#include "ray_trace_scn.h"

#include <algorithm>
#include <numeric>

#include <thrust/transform.h>

#include <array>

using namespace kouek;

void kouek::RayTraceScn::SetModel(std::shared_ptr<Mesh> mesh) {
    this->mesh = mesh;
}

void kouek::RayTraceScn::BuildBVH() {
    std::vector<Mesh::Face2Idx3> bvhFaces;
    bvhFaces.reserve(mesh->GetFS().size());
    std::vector<glm::uint> bvhGrps;
    bvhGrps.reserve(mesh->GetGS().size() - 1);

    const auto &vs = mesh->GetVS();
    const auto &gs = mesh->GetGS();
    const auto &fs = mesh->GetFS();

    AABBBox rootAABB;
    AABBBox cntrRootAABB;

    std::vector<glm::vec3> cntrs;
    std::vector<AABBBox> aabbs, grpAABBs;
    std::vector<glm::uint> indices;
    std::vector<BVHNode> bvh;
    aabbs.reserve(fs.size());
    grpAABBs.reserve(gs.size());
    cntrs.reserve(fs.size());

    bvhHeight = 0;

    struct Param {
        glm::uint start;
        glm::uint num;
        glm::uint lvl;
        glm::uint par;
        bool isLft;
        AABBBox coverAABB;
        AABBBox cntrAABB;
    };
    struct SAHSplit {
        uint8_t dim;
        float split;
        float sah;
    };
    auto findSAHSplit = [&](const Param &param) -> SAHSplit {
        static constexpr uint8_t BinNum = 128;
        static constexpr uint8_t MaxBinIdx = BinNum - 1;
        static constexpr float TraversalCost = 2.f;

        SAHSplit split;
        split.dim = 0;
        split.split = std::numeric_limits<float>::quiet_NaN();

        auto cntrRng = param.cntrAABB.max - param.cntrAABB.min;
        if (glm::dot(cntrRng, cntrRng) <= std::numeric_limits<float>::epsilon())
            return split;

        struct Bin {
            AABBBox aabb = AABBBox::Create();
            glm::uint cnt = 0;
        };
        std::vector<Bin> bins;

        auto invArea = 1.f / param.coverAABB.GetSurfArea();
        auto sah = std::numeric_limits<float>::max();
        uint8_t splitBinIdx = BinNum;

        for (uint8_t xyz = 0; xyz < 3; ++xyz) {
            bins.assign(BinNum, Bin{});

            if (cntrRng[xyz] <= std::numeric_limits<float>::epsilon())
                continue;
            auto invCntrRngDim = 1.f / cntrRng[xyz];

            // Calc the histogram of centerioids
            for (auto fi = param.start; fi < param.start + param.num; ++fi) {
                auto binIdx = (uint8_t)std::min(
                    (float)BinNum *
                        ((cntrs[indices[fi]][xyz] - param.cntrAABB.min[xyz]) *
                         invCntrRngDim),
                    (float)MaxBinIdx);

                bins[binIdx].aabb.Expand(aabbs[indices[fi]]);
                ++bins[binIdx].cnt;
            }

            // Calc right to left AABBs
            std::vector<AABBBox> rhtAABBs;
            rhtAABBs.resize(BinNum);
            auto rhtAABB = AABBBox::Create();
            for (uint8_t binIdxPlus1 = BinNum; binIdxPlus1 > 0; --binIdxPlus1) {
                rhtAABB.Expand(bins[binIdxPlus1 - 1].aabb);
                rhtAABBs[binIdxPlus1 - 1] = rhtAABB;
            }

            // Find best SAH ss point, i is the candidate ss (ss
            // between i and i + 1)
            auto sahTmp = 0.f;
            split.sah = std::numeric_limits<float>::max();
            auto lftAABB = AABBBox::Create();
            glm::uint lftCnt = 0;
            glm::uint rhtCnt = param.num;
            for (uint8_t i = 0; i < MaxBinIdx; ++i) {
                lftAABB.Expand(bins[i].aabb);
                lftCnt += bins[i].cnt;
                rhtCnt -= bins[i].cnt;

                sahTmp = TraversalCost + (lftCnt * lftAABB.GetSurfArea() +
                                          rhtCnt * rhtAABBs[i].GetSurfArea()) *
                                             invArea;
                if (sahTmp < sah) {
                    splitBinIdx = i;
                    split.sah = sah = sahTmp;
                    split.dim = xyz;
                }
            }
        }

        if (splitBinIdx != BinNum)
            split.split = param.cntrAABB.min[split.dim] +
                          cntrRng[split.dim] * (splitBinIdx + 1) / BinNum;
        return split;
    };

    bool buildFaceBVH = true;
    glm::uint grpFstFaceIdx;
    std::function<void(const Param &)> buildNode;
    buildNode = [&](const Param &param) {
        bvhHeight = std::max(bvhHeight, param.lvl);

        glm::uint nodeIdx = bvh.size();
        bvh.emplace_back();
        auto &node = bvh.back();
        node.aabb = param.coverAABB;

        if (param.par != NoneIdx)
            bvh[param.par].dat[param.isLft ? 0 : 1] = bvh.size() - 1;

        if (param.num <= LeafMaxFaceNum) {
            node.type = BVHNode::Type::Leaf;
            if (buildFaceBVH) {
                node.dat[0] = bvhFaces.size();
                node.dat[1] = param.num;
                for (glm::uint i = 0; i < param.num; ++i)
                    bvhFaces.emplace_back(
                        fs[grpFstFaceIdx + indices[param.start + i]]);
            } else {
                node.dat[0] = bvhGrps.size() | BVHNode::BVHNodeLeafBitFlag;
                node.dat[1] = param.num;
                for (glm::uint i = 0; i < param.num; ++i)
                    bvhGrps.emplace_back(indices[param.start + i]);
            }
            return;
        }
        auto dim = param.coverAABB.GetMaxDim();
        auto border = param.coverAABB.GetCenter()[dim];
        auto ss = findSAHSplit(param);

        if (ss.sah == ss.sah) {
            dim = ss.dim;
            border = ss.split;
        }

        node.type = BVHNode::Type::NonLeaf;
        auto lftAABB = AABBBox::Create();
        auto rhtAABB = AABBBox::Create();
        auto lftCntrAABB = AABBBox::Create();
        auto rhtCntrAABB = AABBBox::Create();

        auto splitIdx = param.start;
        bool near2far = (param.num + param.start) & 0x1;

        if ((param.cntrAABB.max - param.cntrAABB.min)[dim] >=
            std::numeric_limits<float>::epsilon()) {
            auto lft = param.start;
            auto rht = param.num + param.start;
            if (near2far)
                while (true) {
                    while (lft != rht && cntrs[indices[lft]][dim] < border) {
                        lftAABB.Expand(aabbs[indices[lft]]);
                        lftCntrAABB.Expand(cntrs[indices[lft]]);
                        ++lft;
                    }
                    if (lft == rht--)
                        break;

                    rhtAABB.Expand(aabbs[indices[lft]]);
                    rhtCntrAABB.Expand(cntrs[indices[lft]]);
                    while (lft != rht && cntrs[indices[rht]][dim] >= border) {
                        rhtAABB.Expand(aabbs[indices[rht]]);
                        rhtCntrAABB.Expand(cntrs[indices[rht]]);
                        --rht;
                    }
                    if (lft == rht)
                        break;

                    lftAABB.Expand(aabbs[indices[rht]]);
                    lftCntrAABB.Expand(cntrs[indices[rht]]);

                    std::swap(indices[lft++], indices[rht]);
                }
            else
                while (true) {
                    while (lft != rht && cntrs[indices[lft]][dim] >= border) {
                        lftAABB.Expand(aabbs[indices[lft]]);
                        lftCntrAABB.Expand(cntrs[indices[lft]]);
                        ++lft;
                    }
                    if (lft == rht--)
                        break;

                    rhtAABB.Expand(aabbs[indices[lft]]);
                    rhtCntrAABB.Expand(cntrs[indices[lft]]);
                    while (lft != rht && cntrs[indices[rht]][dim] < border) {
                        rhtAABB.Expand(aabbs[indices[rht]]);
                        rhtCntrAABB.Expand(cntrs[indices[rht]]);
                        --rht;
                    }
                    if (lft == rht)
                        break;

                    lftAABB.Expand(aabbs[indices[rht]]);
                    lftCntrAABB.Expand(cntrs[indices[rht]]);

                    std::swap(indices[lft++], indices[rht]);
                }
            splitIdx = lft;
        }

        if (splitIdx == param.start || splitIdx == param.start + param.num) {
            splitIdx = param.start + (param.num >> 1);
            for (auto i = param.start; i < splitIdx; ++i) {
                lftAABB.Expand(aabbs[indices[i]]);
                lftCntrAABB.Expand(cntrs[indices[i]]);
            }
            for (auto i = splitIdx; i < param.start + param.num; ++i) {
                rhtAABB.Expand(aabbs[indices[i]]);
                rhtCntrAABB.Expand(cntrs[indices[i]]);
            }
        }

        Param newParam{param.start,   splitIdx - param.start,
                       param.lvl + 1, nodeIdx,
                       true,          lftAABB,
                       lftCntrAABB};
        buildNode(newParam);

        newParam = {splitIdx,      param.num - newParam.num,
                    param.lvl + 1, nodeIdx,
                    false,         rhtAABB,
                    rhtCntrAABB};
        buildNode(newParam);
    };

    // Build BVH for faces in each group
    std::vector<std::vector<BVHNode>> faceBVHs;
    faceBVHs.reserve(gs.size());
    glm::uint d_bvhNum = 0;
    for (glm::uint gi = 0; gi < gs.size() - 1; ++gi) {
        grpFstFaceIdx = gs[gi];

        rootAABB.Reset();
        cntrRootAABB.Reset();

        aabbs.clear();
        cntrs.clear();
        indices.clear();
        {
            auto num = gs[gi + 1] - gs[gi];
            aabbs.reserve(num);
            cntrs.reserve(num);
            indices.reserve(num);
        }

        for (auto fi = gs[gi]; fi < gs[gi + 1]; ++fi) {
            auto &vi3 = fs[fi].v;
            aabbs.emplace_back();
            auto &aabb = aabbs.back();
            aabb.Reset();
            for (uint8_t i = 0; i < 3; ++i)
                aabb.Expand(vs[vi3[i]]);

            cntrs.emplace_back(aabb.GetCenter());

            rootAABB.Expand(aabb);
            cntrRootAABB.Expand(cntrs.back());
        }
        grpAABBs.emplace_back(rootAABB);

        bvh.reserve(2 * std::max((size_t)1, (aabbs.size() / LeafMaxFaceNum)) -
                    1);

        indices.resize(aabbs.size());
        std::iota(indices.begin(), indices.end(), 0);

        Param param{0, aabbs.size(), 0, NoneIdx, false, rootAABB, cntrRootAABB};
        buildNode(param);

        d_bvhNum += bvh.size();
        faceBVHs.emplace_back(std::move(bvh));
    }
    d_faces = bvhFaces;

    // Build BVH for groups
    rootAABB.Reset();
    cntrRootAABB.Reset();

    aabbs = std::move(grpAABBs);
    cntrs.clear();
    indices.clear();
    {
        auto num = gs.size() - 1;
        cntrs.reserve(num);
        indices.reserve(num);
    }

    for (glm::uint gi = 0; gi < gs.size() - 1; ++gi) {
        cntrs.emplace_back(aabbs[gi].GetCenter());

        rootAABB.Expand(aabbs[gi]);
        cntrRootAABB.Expand(cntrs.back());
    }

    indices.resize(aabbs.size());
    std::iota(indices.begin(), indices.end(), 0);

    bvh.reserve(2 * std::max((size_t)1, (aabbs.size() / LeafMaxFaceNum)) - 1);
    buildFaceBVH = false;
    Param param{0, aabbs.size(), 0, NoneIdx, false, rootAABB, cntrRootAABB};
    buildNode(param);
    d_groups = bvhGrps;

    d_bvh.clear();
    d_bvhNum += bvh.size();
    d_bvh.reserve(d_bvhNum);
    d_bvh.insert(d_bvh.end(), bvh.begin(), bvh.end());

    // Compute the offsets of face BVHs in flat mem
    std::vector<glm::uint> grp2faceBVHNodeIndices;
    grp2faceBVHNodeIndices.reserve(gs.size());
    d_bvhNum = bvh.size(); // offset by group BVH
    for (auto &faceBVH : faceBVHs) {
        grp2faceBVHNodeIndices.emplace_back(d_bvhNum);

        auto currStart = d_bvh.end();
        d_bvh.insert(currStart, faceBVH.begin(), faceBVH.end());
        thrust::transform(currStart, d_bvh.end(), currStart,
                          [=] __device__(const BVHNode &node) {
                              auto ret = node;
                              if (ret.type == BVHNode::Type::NonLeaf) {
                                  ret.dat[0] += d_bvhNum;
                                  ret.dat[1] += d_bvhNum;
                              }
                              return ret;
                          });
        d_bvhNum += faceBVH.size(); // offset by previous face BVHs
    }
    grp2faceBVHNodeIndices.emplace_back(d_bvhNum);

    d_grp2faceBVHNodeIndices = grp2faceBVHNodeIndices;

    d_positions = mesh->GetVS();
    d_uvs = mesh->GetVTS();
    d_normals = mesh->GetVNS();
    d_mtls = mesh->GetMtls();
    d_grp2mtls = mesh->GetG2Mtls();
}

SceneInfo kouek::RayTraceScn::GetScnInfo() const {
    SceneInfo info;
    info.faceNum = d_faces.size();
    info.bkgrndCol = bkgrndCol;
    info.grp2faceBVHNodeIndices =
        thrust::raw_pointer_cast(d_grp2faceBVHNodeIndices.data());
    info.grp2mtls = thrust::raw_pointer_cast(d_grp2mtls.data());
    info.groups = thrust::raw_pointer_cast(d_groups.data());
    info.bvh = thrust::raw_pointer_cast(d_bvh.data());
    info.positions = thrust::raw_pointer_cast(d_positions.data());
    info.uvs = thrust::raw_pointer_cast(d_uvs.data());
    info.normals = thrust::raw_pointer_cast(d_normals.data());
    info.mtls = thrust::raw_pointer_cast(d_mtls.data());
    info.faces = thrust::raw_pointer_cast(d_faces.data());
    return info;
}
