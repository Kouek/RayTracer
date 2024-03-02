#include <ray_tracer/lbvh.h>
#include <ray_tracer/test_options.h>

#include <bit>
#include <format>
#include <fstream>
#include <iostream>
#include <utility>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <glm/gtc/matrix_transform.hpp>

#include <cuda/algorithm.h>
#include <cuda/helper.h>

kouek::RayTracer::LBVH::~LBVH() {
    clear();

    if (d_deviceDatPtr) {
        KOUEK_CUDA_CHECK(cudaFree(d_deviceDatPtr));
        d_deviceDatPtr = nullptr;
    }
}

template <typename InodesTy, typename LftCommZeroLenFuncTy>
KOUEK_CUDA_HOST_DEV void computeLBVHLinks(InodesTy &inodes, const uint32_t *mortons,
                                          LftCommZeroLenFuncTy outerLftCommZeroLen,
                                          int64_t inodeNum, int64_t inodeIdx) {
    using InternalNode = kouek::RayTracer::LBVH::InternalNode;

    auto lftCommZeroLen = [&](int64_t fi0, int64_t fi1) -> int {
        return outerLftCommZeroLen(mortons, fi0, fi1, inodeNum);
    };

    int d = [&]() {
        auto commLenLft = lftCommZeroLen(inodeIdx, inodeIdx - 1);
        auto commLenRht = lftCommZeroLen(inodeIdx, inodeIdx + 1);
        if (commLenLft <= commLenRht)
            return 1;
        else if (commLenLft > commLenRht)
            return -1;
    }();

    auto lzlCmp = lftCommZeroLen(inodeIdx, inodeIdx - d);
    int64_t iOffsMax = 2;
    while (true) {
        auto tmpIdx = inodeIdx + d * iOffsMax;
        auto lzl = lftCommZeroLen(inodeIdx, tmpIdx);
        if (lzl <= lzlCmp)
            break;

        iOffsMax <<= 1;
    }

    int64_t iOffs = 0;
    while (true) {
        iOffsMax >>= 1;
        if (iOffsMax < 1)
            break;

        auto tmpIdx = inodeIdx + d * (iOffsMax + iOffs);
        auto lzl = lftCommZeroLen(inodeIdx, tmpIdx);
        if (lzl > lzlCmp)
            iOffs += iOffsMax;
    }
    auto iMax = inodeIdx + d * iOffs;
    lzlCmp = lftCommZeroLen(inodeIdx, iMax);

    iOffsMax = iOffs;
    iOffs = 0;
    while (true) {
        iOffsMax = (iOffsMax + 1) >> 1;
        if (iOffsMax < 1)
            break;

        auto tmpIdx = inodeIdx + d * (iOffsMax + iOffs);
        auto lzl = lftCommZeroLen(inodeIdx, tmpIdx);
        if (lzl > lzlCmp)
            iOffs += iOffsMax;

        if (iOffsMax == 1)
            break;
    }
    auto iSplit = inodeIdx + d * iOffs + (d < 0 ? -1 : 0);

    InternalNode inode;
    inode.lft = iSplit;
    if (glm::min(inodeIdx, iMax) == inode.lft)
        inode.lft |= InternalNode::ChildIsLeafMask;
    inode.rht = iSplit + 1;
    if (glm::max(inodeIdx, iMax) == inode.rht)
        inode.rht |= InternalNode::ChildIsLeafMask;
    inodes[inodeIdx] = inode;
}

void kouek::RayTracer::LBVH::BuildFrom(const InputMesh &inputMesh,
                                       const std::filesystem::path &logPath) {
    isComplete = false;

    clear();

    thrust::device_vector<glm::vec3> d_positions(inputMesh.positions);
    thrust::device_vector<glm::vec<3, IndexTy>> d_facePositionIndices(
        inputMesh.facePositionIndices);
    d_faces.resize(d_facePositionIndices.size());
    thrust::sequence(d_faces.begin(), d_faces.end(), IndexTy(0), IndexTy(1));

    {
        auto minPos = thrust::reduce(
            d_positions.begin(), d_positions.end(), glm::vec3(std::numeric_limits<float>::max()),
            [] KOUEK_CUDA_HOST_DEV(const glm::vec3 &a, const glm::vec3 &b) {
                return glm::vec3{glm::min(a.x, b.x), glm::min(a.y, b.y), glm::min(a.z, b.z)};
            });
        auto maxPos = thrust::reduce(
            d_positions.begin(), d_positions.end(), glm::vec3(std::numeric_limits<float>::lowest()),
            [] KOUEK_CUDA_HOST_DEV(const glm::vec3 &a, const glm::vec3 &b) {
                return glm::vec3{glm::max(a.x, b.x), glm::max(a.y, b.y), glm::max(a.z, b.z)};
            });
        deviceDat.rootAABB.Init(minPos, maxPos);
    }

    thrust::device_vector<uint32_t> d_mortons(d_faces.size());
    // Compute d_mortons,
    // then sort d_faces and d_mortons according to d_mortons
    {
        auto invDltPos = deviceDat.rootAABB.maxPos - deviceDat.rootAABB.minPos;
        for (uint8_t xyz = 0; xyz < 3; ++xyz)
            if (glm::abs(invDltPos[xyz]) < std::numeric_limits<float>::epsilon())
                invDltPos[xyz] = 1.f;
        invDltPos = 1.f / invDltPos;

        thrust::for_each(
            thrust::make_counting_iterator(IndexTy(0)),
            thrust::make_counting_iterator(static_cast<IndexTy>(d_faces.size())),
            [mortons = thrust::raw_pointer_cast(d_mortons.data()),
             facePositionIndices = thrust::raw_pointer_cast(d_facePositionIndices.data()),
             positions = thrust::raw_pointer_cast(d_positions.data()), invDltPos,
             minPos = deviceDat.rootAABB.minPos] __device__(IndexTy fi) {
                auto facePosIdx = facePositionIndices[fi];

                auto aabb = AABB::CreateInitial();
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
                for (uint8_t i = 0; i < 3; ++i)
                    aabb.Expand(positions[facePosIdx[i]]);

                auto centroid = .5f * (aabb.minPos + aabb.maxPos);
                centroid = (centroid - minPos) * invDltPos;
                centroid.x = glm::min(glm::max(centroid.x * 1024.f, 0.f), 1023.f);
                centroid.y = glm::min(glm::max(centroid.y * 1024.f, 0.f), 1023.f);
                centroid.z = glm::min(glm::max(centroid.z * 1024.f, 0.f), 1023.f);

                auto expand10BitsTo32 = [](uint32_t v) {
                    v = (v * 0x00010001u) & 0xFF0000FFu;
                    v = (v * 0x00000101u) & 0x0F00F00Fu;
                    v = (v * 0x00000011u) & 0xC30C30C3u;
                    v = (v * 0x00000005u) & 0x49249249u;
                    return v;
                };
                auto codeX = expand10BitsTo32(static_cast<uint32_t>(centroid.x));
                auto codeY = expand10BitsTo32(static_cast<uint32_t>(centroid.y));
                auto codeZ = expand10BitsTo32(static_cast<uint32_t>(centroid.z));

                mortons[fi] = codeX * 4 + codeY * 2 + codeZ;
            });

        thrust::sort_by_key(d_mortons.begin(), d_mortons.end(), d_faces.begin());
    }
    // Compact d_faces of which have the same morton codes into d_compactedFaces
    // Example:
    // <- Sorted Mortons = 0 0 1 2 2
    // -> Gradients = 1 0 1 1 0
    // -> Prefix Sums = 1 1 2 3 3 -> Compact Num = 3
    // <- Indices of Faces = 0 1 2 3 4
    // -> Compacted Faces = 0 2 3
    d_compactedFaces = [&]() {
        auto d_gradients = CUDA::Difference<uint32_t>(d_mortons);
        return CUDA::CompactIndexes(static_cast<IndexTy>(d_faces.size()), d_gradients);
    }();
    // Compact d_mortons and compute d_aabbs
    thrust::device_vector<AABB> d_aabbs(d_compactedFaces.size(), AABB::CreateInitial());
    {
        decltype(d_mortons) d_newMortons(d_compactedFaces.size());
        thrust::for_each(
            thrust::make_counting_iterator(IndexTy(0)),
            thrust::make_counting_iterator(static_cast<IndexTy>(d_compactedFaces.size())),
            [faceNum = static_cast<IndexTy>(d_faces.size()),
             cmpctNum = static_cast<IndexTy>(d_compactedFaces.size()),
             newMortons = thrust::raw_pointer_cast(d_newMortons.data()),
             mortons = thrust::raw_pointer_cast(d_mortons.data()),
             aabbs = thrust::raw_pointer_cast(d_aabbs.data()),
             faces = thrust::raw_pointer_cast(d_faces.data()),
             facePositionIndices = thrust::raw_pointer_cast(d_facePositionIndices.data()),
             positions = thrust::raw_pointer_cast(d_positions.data()),
             compactedFaces =
                 thrust::raw_pointer_cast(d_compactedFaces.data())] __device__(IndexTy cmpctIdx) {
                newMortons[cmpctIdx] = mortons[compactedFaces[cmpctIdx]];

                auto aabb = AABB::CreateInitial();
                auto nextCmpctFi =
                    cmpctIdx == cmpctNum - 1 ? faceNum : compactedFaces[cmpctIdx + 1];
                for (auto fi = compactedFaces[cmpctIdx]; fi < nextCmpctFi; ++fi) {
                    auto face = faces[fi];
                    auto facePosIdx = facePositionIndices[face];
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
                    for (uint8_t i = 0; i < 3; ++i)
                        aabb.Expand(positions[facePosIdx[i]]);
                }
                aabbs[cmpctIdx] = aabb;
            });
        d_mortons = std::move(d_newMortons);
    }

    auto lftCommZeroLen = [] KOUEK_CUDA_HOST_DEV(const uint32_t *mortons, int64_t cmpctIdx0,
                                                 int64_t cmpctIdx1, int64_t inodeNum) {
        if (cmpctIdx0 < 0 || cmpctIdx0 > inodeNum || cmpctIdx1 < 0 || cmpctIdx1 > inodeNum)
            return -1;

        auto x = mortons[cmpctIdx0] ^ mortons[cmpctIdx1];
#ifdef __CUDA_ARCH__
        return __clz(*reinterpret_cast<int *>(&x));
#else
        return std::countl_zero(x);
#endif // __CUDA_ARCH__
    };
    d_internalNodes.resize(d_compactedFaces.size() - 1);

// #define COMPUTE_LBVH_LINKS_IN_CPU
#ifdef COMPUTE_LBVH_LINKS_IN_CPU
    std::vector<decltype(d_compactedFaces)::value_type> compactedFaces(d_faces.size());
    thrust::copy(d_compactedFaces.begin(), d_compactedFaces.end(), compactedFaces.begin());

    std::vector<decltype(d_mortons)::value_type> mortons(d_mortons.size());
    thrust::copy(d_mortons.begin(), d_mortons.end(), mortons.begin());

    std::vector<decltype(d_internalNodes)::value_type> inodes(d_internalNodes.size());
    int64_t inodeIdx = 0;
    std::for_each(inodes.begin(), inodes.end(),
                  [&, inodeNum = static_cast<int64_t>(inodes.size()),
                   lftCommZeroLen](const decltype(d_internalNodes)::value_type &mortonCode) {
                      computeLBVHLinks(inodes, mortons.data(), lftCommZeroLen, inodeNum, inodeIdx);
                      ++inodeIdx;
                  });
    thrust::copy(inodes.begin(), inodes.end(), d_internalNodes.begin());
#else
    thrust::for_each(thrust::make_counting_iterator(int64_t(0)),
                     thrust::make_counting_iterator(static_cast<int64_t>(d_internalNodes.size())),
                     [inodes = thrust::raw_pointer_cast(d_internalNodes.data()),
                      mortons = thrust::raw_pointer_cast(d_mortons.data()),
                      inodeNum = static_cast<int64_t>(d_internalNodes.size()),
                      lftCommZeroLen] __device__(int64_t inodeIdx) {
                         computeLBVHLinks(inodes, mortons, lftCommZeroLen, inodeNum, inodeIdx);
                     });
#endif // COMPUTE_LBVH_LINKS_IN_CPU

    // Re-compute d_aabbs according to d_inodes
    {
        thrust::for_each(
            thrust::make_counting_iterator(IndexTy(0)),
            thrust::make_counting_iterator(static_cast<IndexTy>(d_internalNodes.size())),
            [aabbs = thrust::raw_pointer_cast(d_aabbs.data()),
             inodes = thrust::raw_pointer_cast(d_internalNodes.data()),
             inodeNum = static_cast<IndexTy>(d_internalNodes.size()),
             lftCommZeroLen] __device__(IndexTy inodeIdx) {
                IndexTy stk[30];
                stk[0] = inodeIdx;
                uint8_t stkSz = 1;
                auto push = [&](IndexTy inodeIdx) {
                    stk[stkSz] = inodeIdx;
                    ++stkSz;
                };
                auto pop = [&]() { --stkSz; };
                auto top = [&]() { return stk[stkSz - 1]; };

                auto aabb = AABB::CreateInitial();
                while (stkSz != 0) {
                    auto &node = inodes[top()];
                    pop();

                    if (node.lft & InternalNode::ChildIsLeafMask)
                        aabb.Expand(aabbs[node.lft & InternalNode::LeafIndexMask]);
                    else
                        push(node.lft);
                    if (node.rht & InternalNode::ChildIsLeafMask)
                        aabb.Expand(aabbs[node.rht & InternalNode::LeafIndexMask]);
                    else
                        push(node.rht);
                }
                inodes[inodeIdx].aabb = aabb;
            });

#ifdef TEST_computeInternalNodes
        test_computeInternalNodes(d_internalNodes, d_faces, d_compactedFaces);
#endif // TEST_computeInternalNodes

#ifdef TEST_computeAABBs
        std::cout << std::format(
            "rootAABB computed by thrust::reduce is ({}, {}, {}) -> ({}, {}, {})\n",
            deviceDat.rootAABB.minPos.x, deviceDat.rootAABB.minPos.y, deviceDat.rootAABB.minPos.z,
            deviceDat.rootAABB.maxPos.x, deviceDat.rootAABB.maxPos.y, deviceDat.rootAABB.maxPos.z);
        InternalNode rootNode = d_internalNodes.front();
        deviceDat.rootAABB = rootNode.aabb;
        std::cout << std::format("d_internalNodes[0].aabb is ({}, {}, {}) -> ({}, {}, {})\n",
                                 deviceDat.rootAABB.minPos.x, deviceDat.rootAABB.minPos.y,
                                 deviceDat.rootAABB.minPos.z, deviceDat.rootAABB.maxPos.x,
                                 deviceDat.rootAABB.maxPos.y, deviceDat.rootAABB.maxPos.z);
#endif // TEST_computeAABBs
    }

    deviceDat.faceNum = d_faces.size();
    deviceDat.compactedFaceNum = d_compactedFaces.size();
    deviceDat.faces = thrust::raw_pointer_cast(d_faces.data());
    deviceDat.compactedFaces = thrust::raw_pointer_cast(d_compactedFaces.data());
    deviceDat.internalNodes = thrust::raw_pointer_cast(d_internalNodes.data());

    uploadDeviceData();

    std::cout << std::format("{} at {}:{}. Builded LBVH.\n", StatTag, __FILE__, __LINE__);
    std::cout << std::format("\tcompact num: {}, face num: {}\n", deviceDat.compactedFaceNum,
                             deviceDat.faceNum);
    isComplete = true;

    if (logPath.empty())
        return;
    std::ofstream os(logPath, std::ios::out);
    if (!os.is_open()) {
        std::cerr << std::format("{} at {}:{}. Cannot open log file at {}.\n", ErrTag, __FILE__,
                                 __LINE__, logPath.string());
        return;
    }

    os << "[OBJ]\n";
    for (IndexTy i = 0; i < inputMesh.facePositionIndices.size(); ++i)
        os << std::format(
            "of:{}-> v0:{}, v1:{}, v2:{}\n", i, inputMesh.facePositionIndices[i][0] + 1,
            inputMesh.facePositionIndices[i][1] + 1, inputMesh.facePositionIndices[i][2] + 1);
    os << "\n[LBVH/Compacts]\n";
    {
        std::vector<IndexTy> compacts(d_compactedFaces.size());
        thrust::copy(d_compactedFaces.begin(), d_compactedFaces.end(), compacts.begin());
        std::vector<IndexTy> faces(d_faces.size());
        thrust::copy(d_faces.begin(), d_faces.end(), faces.begin());
        for (IndexTy i = 0; i < compacts.size(); ++i) {
            auto fi = compacts[i];
            auto fiEnd = i == compacts.size() - 1 ? faces.size() - 1 : compacts[i + 1];

            os << std::format("c:{}-> ", i);
            while (fi < fiEnd) {
                if (fi == fiEnd - 1)
                    os << std::format("f:{}->of:{}", fi, faces[fi]);
                else
                    os << std::format("f:{}->of:{}, ", fi, faces[fi]);
                ++fi;
            }
            os << "\n";
        }
    }
    os << "\n[LBVH/INodes]\n";
    {
        std::vector<InternalNode> inodes(d_internalNodes.size());
        thrust::copy(d_internalNodes.begin(), d_internalNodes.end(), inodes.begin());
        for (IndexTy i = 0; i < inodes.size(); ++i) {
            auto lIsLeaf = (inodes[i].lft & InternalNode::ChildIsLeafMask) != 0;
            auto rIsLeaf = (inodes[i].rht & InternalNode::ChildIsLeafMask) != 0;
            os << std::format("i:{}-> lc:{}{}, rc:{}{}, aabb-> min:({},{},{}), max:({},{},{})\n", i,
                              lIsLeaf ? inodes[i].lft & InternalNode::LeafIndexMask : inodes[i].lft,
                              lIsLeaf ? "L" : "I",
                              rIsLeaf ? inodes[i].rht & InternalNode::LeafIndexMask : inodes[i].rht,
                              rIsLeaf ? "L" : "I", inodes[i].aabb.minPos.x, inodes[i].aabb.minPos.y,
                              inodes[i].aabb.minPos.z, inodes[i].aabb.maxPos.x,
                              inodes[i].aabb.maxPos.y, inodes[i].aabb.maxPos.z);
        }
    }

    os.close();
    std::cout << std::format("{} at {}:{}. Log LBVH to {}.\n", StatTag, __FILE__, __LINE__,
                             logPath.string());
}

void kouek::RayTracer::LBVH::clear() {
    auto release = [](auto &&d_v) {
        d_v.clear();
        d_v.shrink_to_fit();
    };
    release(d_faces);
    release(d_compactedFaces);
    release(d_internalNodes);
}

void kouek::RayTracer::LBVH::uploadDeviceData() {
    if (!d_deviceDatPtr)
        KOUEK_CUDA_CHECK(cudaMalloc(&d_deviceDatPtr, sizeof(*d_deviceDatPtr)));
    KOUEK_CUDA_CHECK(
        cudaMemcpy(d_deviceDatPtr, &deviceDat, sizeof(deviceDat), cudaMemcpyHostToDevice));
}
