#include <ray_caster/dpbx_vdb.h>
#include <ray_caster/test_options.h>

#include <iostream>
#include <numeric>

#include <array>

#include <thrust/binary_search.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>

#include <cuda/algorithm.h>
#include <cuda/helper.h>

#define ASSERT(cond)                                                                               \
    if (!(cond)) {                                                                                 \
        std::cerr << std::format("{} at {}:{}. {} should be true.\n", ErrTag, __FILE__, __LINE__,  \
                                 #cond);                                                           \
        return;                                                                                    \
    }

kouek::RayCaster::DepthBoxVDB::~DepthBoxVDB() {
    clear();

    if (d_deviceDatPtr) {
        KOUEK_CUDA_CHECK(cudaFree(d_deviceDatPtr));
        d_deviceDatPtr = nullptr;
    }
}

struct LevPos {
    uint8_t lev;
    kouek::RayCaster::DepthBoxVDB::CoordTy pos;

    KOUEK_CUDA_HOST_DEV static constexpr LevPos CreateInvalid() {
        return {.lev = kouek::RayCaster::DepthBoxVDB::MaxLevelNum};
    }
    KOUEK_CUDA_HOST_DEV bool operator<(const LevPos &other) const {
        if (lev == other.lev)
            if (pos.x == other.pos.x)
                if (pos.y == other.pos.y)
                    return pos.z < other.pos.z;
                else
                    return pos.y < other.pos.y;
            else
                return pos.x < other.pos.x;
        else
            return lev < other.lev;
    }
    KOUEK_CUDA_HOST_DEV bool operator==(const LevPos &other) const {
        return lev == other.lev && pos == other.pos;
    }
};

KOUEK_CUDA_HOST_DEV inline void
assignNodes(uint8_t lev, kouek::RayCaster::DepthBoxVDB::RelativeIndexTy nodeIdxRelative,
            kouek::RayCaster::DepthBoxVDB::CoordValTy leafNodePerVolYxX,
            kouek::RayCaster::DepthBoxVDB::CoordValTy lx,
            const kouek::RayCaster::DepthBoxVDB::DeviceData &deviceDat,
            const LevPos *compactedLevPoss) {
    using namespace kouek::RayCaster;

    auto &vdbParam = deviceDat.vdbParam;
    DepthBoxVDB::Node inode;
    inode.pos = compactedLevPoss[deviceDat.vdbParam.nodeStarts[lev] + nodeIdxRelative].pos;
    if (lev == 0) {
        inode.childPoolOffsRelative = DepthBoxVDB::UndefRltIdx;
        inode.brickPosInAtlas.z = nodeIdxRelative / leafNodePerVolYxX;
        inode.brickPosInAtlas.y =
            (nodeIdxRelative - inode.brickPosInAtlas.z * leafNodePerVolYxX) / lx;
        inode.brickPosInAtlas.x = nodeIdxRelative - inode.brickPosInAtlas.y * lx -
                                  inode.brickPosInAtlas.z * leafNodePerVolYxX;

        deviceDat.atlasBrickToNodeIndices[inode.brickPosInAtlas.z * vdbParam.atlasBrickPerVDB.y *
                                              vdbParam.atlasBrickPerVDB.x +
                                          inode.brickPosInAtlas.y * vdbParam.atlasBrickPerVDB.x +
                                          inode.brickPosInAtlas.x] = nodeIdxRelative;
    } else
        inode.childPoolOffsRelative =
            static_cast<kouek::RayCaster::DepthBoxVDB::IndexTy>(nodeIdxRelative) *
            vdbParam.dims[lev] * vdbParam.dims[lev] * vdbParam.dims[lev];

    deviceDat.GetNode(lev, nodeIdxRelative) = inode;
}

KOUEK_CUDA_HOST_DEV inline void
assignChildren(uint8_t lev, kouek::RayCaster::DepthBoxVDB::RelativeIndexTy nodeIdxRelative,
               const kouek::RayCaster::DepthBoxVDB::DeviceData &deviceDat) {
    auto &vdbParam = deviceDat.vdbParam;
    auto node = deviceDat.GetNode(lev, nodeIdxRelative);

    auto parLev = vdbParam.rootLev;
    auto par = deviceDat.GetRoot();
    while (parLev > lev + 1) {
        auto cover = [&]() {
            kouek::RayCaster::DepthBoxVDB::CoordValTy cover = 1;
            auto upLev = lev + 1;
            while (upLev < vdbParam.rootLev) {
                cover *= vdbParam.dims[upLev];
                ++upLev;
            }
            return cover;
        }();
        auto pos = node.pos / cover;
        par = deviceDat.GetNode(parLev - 1,
                                deviceDat.GetChild(parLev, par.childPoolOffsRelative, pos));
        --parLev;
    }

    node.pos = node.pos - par.pos * vdbParam.dims[parLev];
    deviceDat.childPools[vdbParam.childStarts[parLev - 1] + par.childPoolOffsRelative +
                         deviceDat.PosRelativeToIdxRelative(parLev, node.pos)] = nodeIdxRelative;
}

template <typename T>
void updateAtlas(std::shared_ptr<kouek::CUDA::Array> &atlasArr,
                 std::unique_ptr<kouek::CUDA::Surface> &atlasSurf,
                 std::unique_ptr<kouek::CUDA::Texture> &atlasTex,
                 std::unique_ptr<kouek::CUDA::Texture> &atlasDepTex,
                 kouek::RayCaster::DepthBoxVDB::DeviceData &deviceDat,
                 const kouek::CUDA::Texture &volTex,
                 const thrust::device_vector<glm::vec<2, T>> &d_emptyScalarRngs) {
    using namespace kouek::CUDA;

    auto &vdbParam = deviceDat.vdbParam;

    kouek::RayCaster::DepthBoxVDB::CoordTy atlasDim = {
        vdbParam.voxPerAtlasBrick * vdbParam.atlasBrickPerVDB.x,
        vdbParam.voxPerAtlasBrick * vdbParam.atlasBrickPerVDB.y,
        vdbParam.voxPerAtlasBrick * vdbParam.atlasBrickPerVDB.z};
    auto atlasChnDesc = [&]() {
        cudaChannelFormatDesc chnDesc;
        cudaExtent extent;
        KOUEK_CUDA_CHECK(cudaArrayGetInfo(&chnDesc, &extent, nullptr, volTex.GetArray()->Get()));
        return chnDesc;
    }();
    if (atlasArr) {
        cudaChannelFormatDesc chnDesc;
        cudaExtent extent;
        KOUEK_CUDA_CHECK(cudaArrayGetInfo(&chnDesc, &extent, nullptr, atlasArr->Get()));

        if (chnDesc.x != atlasChnDesc.x || chnDesc.y != atlasChnDesc.y ||
            chnDesc.z != atlasChnDesc.z || chnDesc.w != atlasChnDesc.w ||
            chnDesc.f != atlasChnDesc.f || extent.width != atlasDim.x ||
            extent.height != atlasDim.y || extent.depth != atlasDim.z) {
            atlasSurf.reset();
            atlasTex.reset();
            atlasArr.reset();
        }
    }

    if (!atlasArr) {
        atlasArr = std::make_shared<Array>(atlasChnDesc, atlasDim);
        atlasSurf = std::make_unique<Surface>(atlasArr);
        atlasTex = std::make_unique<Texture>(atlasArr);
        atlasDepTex = std::make_unique<Texture>(atlasArr, []() {
            cudaTextureDesc texDesc;
            memset(&texDesc, 0, sizeof(texDesc));
            texDesc.normalizedCoords = 0;
            texDesc.filterMode = cudaFilterModePoint;
            texDesc.addressMode[0] = texDesc.addressMode[1] = texDesc.addressMode[2] =
                cudaAddressModeBorder;
            texDesc.readMode = cudaReadModeElementType;
            return texDesc;
        }());

        deviceDat.atlasSurf = atlasSurf->Get();
        deviceDat.atlasTex = atlasTex->Get();
        deviceDat.atlasDepTex = atlasDepTex->Get();
    }

    auto voxNumPerAtlas =
        static_cast<kouek::RayCaster::DepthBoxVDB::IndexTy>(atlasDim.x) * atlasDim.y * atlasDim.z;
    thrust::for_each(
        thrust::make_counting_iterator(kouek::RayCaster::DepthBoxVDB::IndexTy(0)),
        thrust::make_counting_iterator(voxNumPerAtlas),
        [scalarRngNum = static_cast<uint8_t>(d_emptyScalarRngs.size()), volTex = volTex.Get(),
         atlasDimYxX = static_cast<kouek::RayCaster::DepthBoxVDB::IndexTy>(atlasDim.x) * atlasDim.y,
         ax = static_cast<kouek::RayCaster::DepthBoxVDB::IndexTy>(atlasDim.x), deviceDat,
         emptyScalarRngs = thrust::raw_pointer_cast(
             d_emptyScalarRngs.data())] __device__(kouek::RayCaster::DepthBoxVDB::IndexTy voxIdx) {
            auto &vdbParam = deviceDat.vdbParam;

            kouek::RayCaster::DepthBoxVDB::CoordTy voxInAtlas;
            voxInAtlas.z = voxIdx / atlasDimYxX;
            voxInAtlas.y = (voxIdx - voxInAtlas.z * atlasDimYxX) / ax;
            voxInAtlas.x = voxIdx - voxInAtlas.y * ax - voxInAtlas.z * atlasDimYxX;

            auto brickInAtlas = voxInAtlas / vdbParam.voxPerAtlasBrick;
            auto nodeIdx = deviceDat.BrickPosInAtlasToNodeIdxRelative(brickInAtlas);
            if (nodeIdx == kouek::RayCaster::DepthBoxVDB::UndefRltIdx)
                return; // empty atlas brick

            auto minVoxPosInVDB = deviceDat.GetNode(0, nodeIdx).pos;
            auto voxInAtlasBrick = voxInAtlas - brickInAtlas * vdbParam.voxPerAtlasBrick;

            kouek::RayCaster::DepthBoxVDB::CoordTy depDir = {0, 0, 0};
            if (deviceDat.vdbParam.apronDepWid != deviceDat.vdbParam.apronWid)
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
                for (uint8_t xyz = 0; xyz < 3; ++xyz)
                    depDir[xyz] = voxInAtlasBrick[xyz] == 0                               ? 1
                                  : voxInAtlasBrick[xyz] == vdbParam.voxPerAtlasBrick - 1 ? -1
                                                                                          : 0;

            // At the area surrounded by the apron (including the apron),
            // assign the original volume data
            if (depDir.x == 0 && depDir.y == 0 && depDir.z == 0) {
                auto v = tex3D<T>(
                    volTex,
                    static_cast<float>(minVoxPosInVDB.x) + voxInAtlasBrick.x - vdbParam.apronDepWid,
                    static_cast<float>(minVoxPosInVDB.y) + voxInAtlasBrick.y - vdbParam.apronDepWid,
                    static_cast<float>(minVoxPosInVDB.z) + voxInAtlasBrick.z -
                        vdbParam.apronDepWid);
                surf3Dwrite(v, deviceDat.atlasSurf, sizeof(T) * voxInAtlas.x, voxInAtlas.y,
                            voxInAtlas.z);
                return;
            }

            // At the area of depth, but multi depth-directions are assigned, set 0
            if (glm::abs(depDir.x) + glm::abs(depDir.y) + glm::abs(depDir.z) > 1) {
                surf3Dwrite(T(0), deviceDat.atlasSurf, sizeof(T) * voxInAtlas.x, voxInAtlas.y,
                            voxInAtlas.z);
                return;
            }
            // At the area of depth, move along the depth-direction with a 3x3 window
            // to compute the depth
            auto dltDir0 = depDir.x != 0   ? glm::vec3{0.f, 1.f, 0.f}
                           : depDir.y != 0 ? glm::vec3{0.f, 0.f, 1.f}
                                           : glm::vec3{1.f, 0.f, 0.f};
            auto dltDir1 = depDir.x != 0   ? glm::vec3{0.f, 0.f, 1.f}
                           : depDir.y != 0 ? glm::vec3{1.f, 0.f, 0.f}
                                           : glm::vec3{0.f, 1.f, 0.f};
            // Ignore the first 3x3 of the 0-th depth
            auto voxInVDB = glm::vec3(minVoxPosInVDB) +
                            glm::vec3{depDir.x == 1    ? 1.f
                                      : depDir.x == -1 ? vdbParam.dims[0] - 2
                                                       : voxInAtlasBrick.x - vdbParam.apronDepWid,
                                      depDir.y == 1    ? 1.f
                                      : depDir.y == -1 ? vdbParam.dims[0] - 2
                                                       : voxInAtlasBrick.y - vdbParam.apronDepWid,
                                      depDir.z == 1    ? 1.f
                                      : depDir.z == -1 ? vdbParam.dims[0] - 2
                                                       : voxInAtlasBrick.z - vdbParam.apronDepWid};

            kouek::RayCaster::DepthBoxVDB::CoordValTy dep;
            for (dep = 0; dep < vdbParam.dims[0] - 1; ++dep) {
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
                for (int8_t t = 0; t < 9; ++t) {
                    auto dlt0 = (t % 3) - 1.f;
                    auto dlt1 = (t / 3) - 1.f;
                    auto samplePos = voxInVDB + dlt0 * dltDir0 + dlt1 * dltDir1;
                    auto v = tex3D<T>(volTex, samplePos.x, samplePos.y, samplePos.z);

                    bool empty = false;
                    for (uint8_t ri = 0; ri < scalarRngNum; ++ri)
                        empty |= v >= emptyScalarRngs[ri][0] && v <= emptyScalarRngs[ri][1];
                    if (!empty)
                        goto TERMINAL;
                }

                voxInVDB += depDir;
            }
        TERMINAL:
            surf3Dwrite(static_cast<T>(dep), deviceDat.atlasSurf, sizeof(T) * voxInAtlas.x,
                        voxInAtlas.y, voxInAtlas.z);
        });
}

template <typename T>
void kouek::RayCaster::DepthBoxVDB::BuildFrom(
    const InputVolumeAndConfiguration<T> &inputVolAndCfg) {
    isComplete = false;

    clear();

    for (auto log2Dim : inputVolAndCfg.log2Dims)
        ASSERT(log2Dim > 0 && log2Dim <= MaxLog2Dim);

    auto voxPerVol = inputVolAndCfg.volTex.GetArray()->GetExtent();
    ASSERT(voxPerVol.width != 0 && voxPerVol.height != 0 && voxPerVol.depth != 0);

    auto &vdbParam = deviceDat.vdbParam;

    vdbParam.apronWid = inputVolAndCfg.apronWid;
    vdbParam.apronDepWid = vdbParam.apronWid + (inputVolAndCfg.useDepthBox ? 1 : 0);
    for (uint8_t lev = 0; lev < MaxLevelNum; ++lev) {
        vdbParam.log2Dims[lev] = inputVolAndCfg.log2Dims[lev];
        vdbParam.dims[lev] = kouek::RayCaster::DepthBoxVDB::CoordValTy(1) << vdbParam.log2Dims[lev];
    }
    vdbParam.minDepPosValInBrick = -vdbParam.apronDepWid; // 0 - (0 + apronDepWid)
    vdbParam.maxDepPosValInBrick =
        vdbParam.dims[0] - 1 +
        vdbParam.apronDepWid; // (0 + apronDepWid + dims[0] - 1 + apronDepWid) - (0 + apronDepWid)

    // Compute the root level
    glm::vec<3, RelativeIndexTy> leafNodePerVol = {
        (voxPerVol.width + vdbParam.dims[0] - 1) / vdbParam.dims[0],
        (voxPerVol.height + vdbParam.dims[0] - 1) / vdbParam.dims[0],
        (voxPerVol.depth + vdbParam.dims[0] - 1) / vdbParam.dims[0]};
    {
        vdbParam.rootLev = 0;
        vdbParam.voxsPerChild[0] = 1.f;
        kouek::RayCaster::DepthBoxVDB::IndexTy coverVox = vdbParam.dims[0];
        auto check = [&]() {
            return coverVox >= voxPerVol.width && coverVox >= voxPerVol.height &&
                   coverVox >= voxPerVol.depth;
        };

        if (!check()) {
            uint8_t lev;
            for (lev = 1; lev < MaxLevelNum; ++lev) {
                vdbParam.voxsPerChild[lev] = coverVox;
                coverVox *= vdbParam.dims[lev];

                ++vdbParam.rootLev;
                if (check())
                    break;
            }

            if (lev == MaxLevelNum) {
                std::cerr << std::format(
                    "{} at {}:{}. Current VDB configuration cannot cover {}x{}x{} voxels.\n",
                    ErrTag, __FILE__, __LINE__, voxPerVol.width, voxPerVol.height, voxPerVol.depth);
                return;
            }

            vdbParam.voxPerVDB = coverVox;
            vdbParam.voxPerVol.x = voxPerVol.width;
            vdbParam.voxPerVol.y = voxPerVol.height;
            vdbParam.voxPerVol.z = voxPerVol.depth;
        }
    }

    // Downsample each brick of volume data corresponding to each leaves.
    // Judge whther each leaf is empty or not, remove the empty leaves and
    // compact the valid leaves into d_compactedLeaves
    auto leafNodePerVolYxX = leafNodePerVol.x * leafNodePerVol.y;
    thrust::device_vector<glm::vec<2, T>> d_emptyScalarRngs = inputVolAndCfg.emptyScalarRngs;
    auto d_compactedLeaves = [&]() {
        auto leafNum = leafNodePerVolYxX * leafNodePerVol.z;

        thrust::device_vector<kouek::RayCaster::DepthBoxVDB::RelativeIndexTy> d_leafValids(leafNum,
                                                                                           1);

        thrust::for_each(
            thrust::make_counting_iterator(kouek::RayCaster::DepthBoxVDB::RelativeIndexTy(0)),
            thrust::make_counting_iterator(leafNum),
            [scalarRngNum = static_cast<uint8_t>(d_emptyScalarRngs.size()),
             voxPerBrick = vdbParam.dims[0],
             voxNumPerLeaf = vdbParam.dims[0] * vdbParam.dims[0] * vdbParam.dims[0],
             leafNodePerVolYxX, lx = leafNodePerVol.x,
             leafValids = thrust::raw_pointer_cast(d_leafValids.data()),
             emptyScalarRngs = thrust::raw_pointer_cast(d_emptyScalarRngs.data()),
             volTex =
                 inputVolAndCfg.volTex
                     .Get()] __device__(kouek::RayCaster::DepthBoxVDB::RelativeIndexTy leafIdx) {
                kouek::RayCaster::DepthBoxVDB::CoordTy pos;
                pos.z = leafIdx / leafNodePerVolYxX;
                pos.y = (leafIdx - pos.z * leafNodePerVolYxX) / lx;
                pos.x = leafIdx - pos.z * leafNodePerVolYxX - pos.y * lx;
                pos *= voxPerBrick;

                kouek::RayCaster::DepthBoxVDB::RelativeIndexTy emptyVoxNum = 0;
                for (kouek::RayCaster::DepthBoxVDB::RelativeIndexTy z = 0; z < voxPerBrick; ++z)
                    for (kouek::RayCaster::DepthBoxVDB::RelativeIndexTy y = 0; y < voxPerBrick; ++y)
                        for (kouek::RayCaster::DepthBoxVDB::RelativeIndexTy x = 0; x < voxPerBrick;
                             ++x) {
                            auto v = tex3D<T>(volTex, pos.x + x, pos.y + y, pos.z + z);

                            for (uint8_t ri = 0; ri < scalarRngNum; ++ri)
                                if (v >= emptyScalarRngs[ri][0] && v <= emptyScalarRngs[ri][1])
                                    ++emptyVoxNum;
                        }
                if (emptyVoxNum == voxNumPerLeaf)
                    leafValids[leafIdx] = 0;
            });

        return CUDA::CompactIndexes(leafNum, d_leafValids);
    }();

    // Compute atlas space needed
    vdbParam.voxPerAtlasBrick = vdbParam.dims[0] + vdbParam.apronDepWid * 2;
    vdbParam.atlasBrickPerVDB = {leafNodePerVol.x, leafNodePerVol.y, 1};
    while (![&]() {
        auto atlasNumPerVDB =
            vdbParam.atlasBrickPerVDB.x * vdbParam.atlasBrickPerVDB.y * vdbParam.atlasBrickPerVDB.z;
        return atlasNumPerVDB >= d_compactedLeaves.size();
    }())
        ++vdbParam.atlasBrickPerVDB.z;
    d_atlasBrickToNodeIndices.resize(vdbParam.atlasBrickPerVDB.x * vdbParam.atlasBrickPerVDB.y *
                                         vdbParam.atlasBrickPerVDB.z,
                                     UndefRltIdx);
    deviceDat.atlasBrickToNodeIndices = thrust::raw_pointer_cast(d_atlasBrickToNodeIndices.data());

    // Compute d_levPoss,
    // sort d_levPoss and find its right boundary and
    // compact d_levPoss into d_compactedLevPoss
    auto d_compactedLevPoss = [&]() {
        thrust::device_vector<LevPos> d_levPoss(MaxLevelNum * d_compactedLeaves.size(),
                                                LevPos::CreateInvalid());
        thrust::for_each(
            thrust::make_counting_iterator(kouek::RayCaster::DepthBoxVDB::RelativeIndexTy(0)),
            thrust::make_counting_iterator(
                static_cast<kouek::RayCaster::DepthBoxVDB::RelativeIndexTy>(
                    d_compactedLeaves.size())),
            [leafNodePerVolYxX, lx = leafNodePerVol.x, cmpctNum = d_compactedLeaves.size(),
             vdbParam = deviceDat.vdbParam,
             levPosPools = thrust::raw_pointer_cast(d_levPoss.data()),
             compactedLeaves = thrust::raw_pointer_cast(
                 d_compactedLeaves
                     .data())] __device__(kouek::RayCaster::DepthBoxVDB::RelativeIndexTy cmpctIdx) {
                auto leafIdx = compactedLeaves[cmpctIdx];
                LevPos lp;
                lp.lev = 0;
                lp.pos.z = leafIdx / leafNodePerVolYxX;
                lp.pos.y = (leafIdx - lp.pos.z * leafNodePerVolYxX) / lx;
                lp.pos.x = leafIdx - lp.pos.z * leafNodePerVolYxX - lp.pos.y * lx;

                levPosPools[cmpctIdx] = lp;

                for (uint8_t lev = 1; lev <= vdbParam.rootLev; ++lev) {
                    lp.pos = lp.pos / vdbParam.dims[lev];
                    lp.lev = lev;
                    levPosPools[lev * cmpctNum + cmpctIdx] = lp;
                }
            });

        thrust::sort(d_levPoss.begin(), d_levPoss.end());

        kouek::RayCaster::DepthBoxVDB::RelativeIndexTy validLevPosNum = [&]() {
            auto last =
                thrust::lower_bound(d_levPoss.begin(), d_levPoss.end(), LevPos::CreateInvalid());
            return thrust::distance(d_levPoss.begin(), last);
        }();

        auto d_diffs = CUDA::Difference<kouek::RayCaster::DepthBoxVDB::RelativeIndexTy>(
            d_levPoss, validLevPosNum);
        return CUDA::Compact(d_levPoss, d_diffs, validLevPosNum);
    }();

    // Compute d_nodePools and d_childPools
    std::array<kouek::RayCaster::DepthBoxVDB::RelativeIndexTy, MaxLevelNum> nodeNums;
    kouek::RayCaster::DepthBoxVDB::IndexTy totNodeNum = 0;
    kouek::RayCaster::DepthBoxVDB::IndexTy totChildNum = 0;
    auto currPtr = d_compactedLevPoss.begin();
    for (uint8_t lev = 0; lev <= vdbParam.rootLev; ++lev) {
        LevPos nextLevMin = {lev + 1, {0, 0, 0}};
        auto nextPtr = thrust::lower_bound(currPtr, d_compactedLevPoss.end(), nextLevMin);

        nodeNums[lev] = thrust::distance(currPtr, nextPtr);
        vdbParam.nodeStarts[lev] = totNodeNum;
        totNodeNum += nodeNums[lev];
        currPtr = nextPtr;

        if (lev != 0) {
            auto childNum = static_cast<kouek::RayCaster::DepthBoxVDB::IndexTy>(nodeNums[lev]) *
                            vdbParam.dims[lev] * vdbParam.dims[lev] * vdbParam.dims[lev];
            vdbParam.childStarts[lev - 1] = totChildNum;
            totChildNum += childNum;
        }
    }

// #define COMPUTE_VDB_LINKS_IN_CPU
#ifdef COMPUTE_VDB_LINKS_IN_CPU
    // Assign nodes
    std::vector<decltype(d_nodePools)::value_type> nodePools(totNodeNum);

    std::vector<decltype(d_atlasBrickToNodeIndices)::value_type> atlasBrickToNodeIndices(
        d_atlasBrickToNodeIndices.size());
    thrust::copy(d_atlasBrickToNodeIndices.begin(), d_atlasBrickToNodeIndices.end(),
                 atlasBrickToNodeIndices.begin());
    std::vector<LevPos> compactedLevPoss(d_compactedLevPoss.size());
    thrust::copy(d_compactedLevPoss.begin(), d_compactedLevPoss.end(), compactedLevPoss.begin());

    deviceDat.nodePools = nodePools.data();
    deviceDat.atlasBrickToNodeIndices = atlasBrickToNodeIndices.data();
    for (uint8_t lev = 0; lev < vdbParam.rootLev; ++lev)
        for (RelativeIndexTy niRlt = 0; niRlt < nodeNums[lev]; ++niRlt)
            assignNodes(lev, niRlt, leafNodePerVolYxX, leafNodePerVol.x, deviceDat,
                        compactedLevPoss.data());
    {
        Node rootNode;
        rootNode.pos = {0, 0, 0};
        rootNode.childPoolOffsRelative = 0;
        nodePools[totNodeNum - 1] = rootNode;
    }

    // Assign parent and child relations
    std::vector<decltype(d_childPools)::value_type> childPools(totChildNum, UndefRltIdx);
    deviceDat.childPools = childPools.data();
    for (uint8_t compoLev = 0; compoLev < vdbParam.rootLev; ++compoLev) {
        auto lev = vdbParam.rootLev - 1 - compoLev;
        for (RelativeIndexTy niRlt = 0; niRlt < nodeNums[lev]; ++niRlt)
            assignChildren(lev, niRlt, deviceDat);
    }

    // Transform node.pos from position-to-parent-level to position-to-volume
    for (uint8_t lev = 0; lev < vdbParam.rootLev; ++lev)
        for (RelativeIndexTy niRlt = 0; niRlt < nodeNums[lev]; ++niRlt) {
            auto pos = deviceDat.GetNode(lev, niRlt).pos;
            for (auto l = lev + 1; l > 0; --l)
                pos *= deviceDat.vdbParam.dims[l - 1];
            deviceDat.GetNode(lev, niRlt).pos = pos;
        }

    d_nodePools.resize(totNodeNum);
    thrust::copy(nodePools.begin(), nodePools.end(), d_nodePools.begin());
    d_childPools.resize(totChildNum);
    thrust::copy(childPools.begin(), childPools.end(), d_childPools.begin());
    d_atlasBrickToNodeIndices.resize(atlasBrickToNodeIndices.size());
    thrust::copy(atlasBrickToNodeIndices.begin(), atlasBrickToNodeIndices.end(),
                 d_atlasBrickToNodeIndices.begin());

    deviceDat.nodePools = thrust::raw_pointer_cast(d_nodePools.data());
    deviceDat.childPools = thrust::raw_pointer_cast(d_childPools.data());
    deviceDat.atlasBrickToNodeIndices = thrust::raw_pointer_cast(d_atlasBrickToNodeIndices.data());
#else
    // Assign nodes
    d_nodePools.resize(totNodeNum);
    deviceDat.nodePools = thrust::raw_pointer_cast(d_nodePools.data());
    for (uint8_t lev = 0; lev < vdbParam.rootLev; ++lev)
        thrust::for_each(
            thrust::make_counting_iterator(kouek::RayCaster::DepthBoxVDB::RelativeIndexTy(0)),
            thrust::make_counting_iterator(nodeNums[lev]),
            [lev, leafNodePerVolYxX, lx = leafNodePerVol.x, deviceDat = this->deviceDat,
             compactedLevPoss = thrust::raw_pointer_cast(
                 d_compactedLevPoss
                     .data())] __device__(kouek::RayCaster::DepthBoxVDB::RelativeIndexTy
                                              nodeIdxRelative) {
                assignNodes(lev, nodeIdxRelative, leafNodePerVolYxX, lx, deviceDat,
                            compactedLevPoss);
            });
    {
        Node rootNode;
        rootNode.pos = {0, 0, 0};
        rootNode.childPoolOffsRelative = 0;
        d_nodePools[totNodeNum - 1] = rootNode;
    }

    // Assign parent and child relations
    d_childPools.resize(totChildNum, UndefRltIdx);
    deviceDat.childPools = thrust::raw_pointer_cast(d_childPools.data());
    for (uint8_t compoLev = 0; compoLev < vdbParam.rootLev; ++compoLev) {
        auto lev = vdbParam.rootLev - 1 - compoLev;
        thrust::for_each(
            thrust::make_counting_iterator(kouek::RayCaster::DepthBoxVDB::RelativeIndexTy(0)),
            thrust::make_counting_iterator(nodeNums[lev]),
            [lev, deviceDat = this->deviceDat] __device__(
                kouek::RayCaster::DepthBoxVDB::RelativeIndexTy nodeIdxRelative) {
                assignChildren(lev, nodeIdxRelative, deviceDat);
            });
    }

    // Transform node.pos from position-to-parent-level to position-to-volume
    for (uint8_t lev = 0; lev < vdbParam.rootLev; ++lev)
        thrust::for_each(
            thrust::make_counting_iterator(kouek::RayCaster::DepthBoxVDB::RelativeIndexTy(0)),
            thrust::make_counting_iterator(nodeNums[lev]),
            [lev, deviceDat = this->deviceDat] __device__(
                kouek::RayCaster::DepthBoxVDB::RelativeIndexTy nodeIdxRelative) {
                auto pos = deviceDat.GetNode(lev, nodeIdxRelative).pos;
                for (auto l = lev + 1; l > 0; --l)
                    pos *= deviceDat.vdbParam.dims[l - 1];
                deviceDat.GetNode(lev, nodeIdxRelative).pos = pos;
            });
#endif // COMPUTE_VDB_LINKS_IN_CPU

#ifdef TEST_computeNodesAndChildren
    test_computeNodesAndChildren(d_compactedLeaves.size(), deviceDat, d_nodePools, d_childPools);
#endif // TEST_computeNodesAndChildren

    updateAtlas<T>(atlasArr, atlasSurf, atlasTex, atlasDepTex, deviceDat, inputVolAndCfg.volTex,
                   d_emptyScalarRngs);
    uploadDeviceData();

#ifdef TEST_rayCastVDB
    test_rayCastVDB(deviceDat, d_nodePools, d_childPools, d_atlasBrickToNodeIndices);
#endif // TEST_rayCastVDB

    // Statistics
    std::cout << "Build VDB with\n";
    size_t memUsed = 0;
    std::cout << std::format("\tNode pools of size {}x{}\n", d_nodePools.size(),
                             sizeof(kouek::RayCaster::DepthBoxVDB::Node));
    memUsed += d_nodePools.size() * sizeof(kouek::RayCaster::DepthBoxVDB::Node);
    std::cout << std::format("\tChild pools of size {}x{}\n", d_childPools.size(),
                             sizeof(*kouek::RayCaster::DepthBoxVDB::DeviceData::childPools));
    memUsed += d_childPools.size() * sizeof(*kouek::RayCaster::DepthBoxVDB::DeviceData::childPools);
    std::cout << std::format(
        "\tMaps from atlas brick to node index of size {}x{}\n", d_atlasBrickToNodeIndices.size(),
        sizeof(*kouek::RayCaster::DepthBoxVDB::DeviceData::atlasBrickToNodeIndices));
    memUsed += d_atlasBrickToNodeIndices.size() *
               sizeof(*kouek::RayCaster::DepthBoxVDB::DeviceData::atlasBrickToNodeIndices);
    std::cout << std::format("\tAtlas cudaArray of size {}x{}x{}x{}\n",
                             vdbParam.atlasBrickPerVDB.x * vdbParam.voxPerAtlasBrick,
                             vdbParam.atlasBrickPerVDB.y * vdbParam.voxPerAtlasBrick,
                             vdbParam.atlasBrickPerVDB.z * vdbParam.voxPerAtlasBrick, sizeof(T));
    memUsed += sizeof(T) * vdbParam.atlasBrickPerVDB.x * vdbParam.voxPerAtlasBrick *
               vdbParam.atlasBrickPerVDB.y * vdbParam.voxPerAtlasBrick *
               vdbParam.atlasBrickPerVDB.z * vdbParam.voxPerAtlasBrick;
    std::cout << std::format("\tTotal use of memory {}\n", memUsed);

    isComplete = true;
}

template void
kouek::RayCaster::DepthBoxVDB::BuildFrom(const InputVolumeAndConfiguration<uint8_t> &);
template void
kouek::RayCaster::DepthBoxVDB::BuildFrom(const InputVolumeAndConfiguration<uint16_t> &);

size_t kouek::RayCaster::DepthBoxVDB::GetNodePoolsSize() const { return d_nodePools.size(); }
size_t kouek::RayCaster::DepthBoxVDB::GetChildPoolsSize() const { return d_childPools.size(); }
size_t kouek::RayCaster::DepthBoxVDB::GetAtlasBrickToNodeIndicesSize() const {
    return d_atlasBrickToNodeIndices.size();
}

void kouek::RayCaster::DepthBoxVDB::clear() {
    d_nodePools.clear();
    d_nodePools.shrink_to_fit();
    d_childPools.clear();
    d_childPools.shrink_to_fit();
    d_atlasBrickToNodeIndices.clear();
    d_atlasBrickToNodeIndices.shrink_to_fit();
}

void kouek::RayCaster::DepthBoxVDB::uploadDeviceData() {
    if (!d_deviceDatPtr)
        KOUEK_CUDA_CHECK(cudaMalloc(&d_deviceDatPtr, sizeof(*d_deviceDatPtr)));
    KOUEK_CUDA_CHECK(
        cudaMemcpy(d_deviceDatPtr, &deviceDat, sizeof(deviceDat), cudaMemcpyHostToDevice));
}
