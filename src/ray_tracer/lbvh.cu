#include <ray_tracer/lbvh.h>
#include <ray_tracer/test_options.h>

#include <bit>
#include <utility>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <glm/gtc/matrix_transform.hpp>

#include <cuda/algorithm.h>
#include <cuda/helper.h>

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
    if (inodeIdx == 90)
        int x = 0;
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

void kouek::RayTracer::LBVH::BuildFrom(const InputMesh &mesh) {
    isComplete = false;

    thrust::device_vector<glm::vec3> d_positions(mesh.positions);
    thrust::device_vector<glm::vec<3, IndexTy>> d_facePositionIndices(mesh.facePositionIndices);
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
        rootAABB.Init(minPos, maxPos);
    }

    thrust::device_vector<uint32_t> d_mortons(d_faces.size());
    d_aabbs.clear();
    // Initialize d_aabbs
    d_aabbs.assign(d_faces.size(), AABB::CreateInitial());
    // Compute d_aabbs and d_mortons,
    // then sort d_faces and d_mortons according to d_mortons
    {
        auto invDltPos = rootAABB.maxPos - rootAABB.minPos;
        for (uint8_t xyz = 0; xyz < 3; ++xyz)
            if (glm::abs(invDltPos[xyz]) < std::numeric_limits<float>::epsilon())
                invDltPos[xyz] = 1.f;
        invDltPos = 1.f / invDltPos;

        thrust::for_each(
            thrust::make_counting_iterator(IndexTy(0)),
            thrust::make_counting_iterator(static_cast<IndexTy>(d_faces.size())),
            [aabbs = thrust::raw_pointer_cast(d_aabbs.data()),
             mortons = thrust::raw_pointer_cast(d_mortons.data()),
             facePositionIndices = thrust::raw_pointer_cast(d_facePositionIndices.data()),
             positions = thrust::raw_pointer_cast(d_positions.data()), invDltPos,
             minPos = rootAABB.minPos] __device__(IndexTy fi) {
                auto facePosIdx3 = facePositionIndices[fi];

                auto aabb = AABB::CreateInitial();
#pragma unroll
                for (uint8_t i = 0; i < 3; ++i)
                    aabb.Expand(positions[facePosIdx3[i]]);
                aabbs[fi] = aabb;

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
    // <- Faces = 0 1 2 3 4
    // -> Gradients = 1 0 1 1 0
    // -> Prefix Sums = 1 1 2 3 3 -> Compact Num = 3
    // -> Compacted Faces = 0 2 3
    auto d_compactedFaces = [&]() {
        auto d_gradients = CUDA::Difference<uint32_t>(d_mortons);
        return CUDA::CompactIndexes(static_cast<IndexTy>(d_faces.size()), d_gradients);
    }();
    // Compact d_mortons
    {
        decltype(d_mortons) d_newMortons(d_compactedFaces.size());
        thrust::for_each(
            thrust::make_counting_iterator(IndexTy(0)),
            thrust::make_counting_iterator(static_cast<IndexTy>(d_compactedFaces.size())),
            [newMortons = thrust::raw_pointer_cast(d_newMortons.data()),
             mortons = thrust::raw_pointer_cast(d_mortons.data()),
             compactedFaces =
                 thrust::raw_pointer_cast(d_compactedFaces.data())] __device__(IndexTy cmpctIdx) {
                newMortons[cmpctIdx] = mortons[compactedFaces[cmpctIdx]];
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

#ifdef TEST_computeInternalNodes
    test_computeInternalNodes(d_internalNodes);
    return;
#endif // TEST_computeInternalNodes

    // Re-compute d_aabbs according to d_inodes
    {
        decltype(d_aabbs) d_newAABBs;
        d_newAABBs.assign(d_internalNodes.size(), AABB::CreateInitial());
        thrust::for_each(
            thrust::make_counting_iterator(IndexTy(0)),
            thrust::make_counting_iterator(static_cast<IndexTy>(d_internalNodes.size())),
            [newAABBs = thrust::raw_pointer_cast(d_newAABBs.data()),
             aabbs = thrust::raw_pointer_cast(d_aabbs.data()),
             inodes = thrust::raw_pointer_cast(d_internalNodes.data()),
             inodeNum = static_cast<IndexTy>(d_internalNodes.size()),
             lftCommZeroLen] __device__(IndexTy inodeIdx) {
                IndexTy stk[30];
                uint8_t stkSz = 1;
                auto push = [&](IndexTy inodeIdx) {
                    stk[stkSz] = inodeIdx;
                    ++stkSz;
                };
                auto pop = [&]() { --stkSz; };
                auto top = [&]() { return stk[stkSz - 1]; };

                stk[0] = inodeIdx;
                auto newAABB = AABB::CreateInitial();
                while (stkSz != 0) {
                    auto &node = inodes[top()];
                    pop();

                    if (node.lft & InternalNode::ChildIsLeafMask)
                        newAABB.Expand(aabbs[node.lft & InternalNode::LeafIndexMask]);
                    else
                        push(node.lft);
                    if (node.rht & InternalNode::ChildIsLeafMask)
                        newAABB.Expand(aabbs[node.rht & InternalNode::LeafIndexMask]);
                    else
                        push(node.rht);
                }
                newAABBs[inodeIdx] = newAABB;
            });
        d_aabbs = std::move(d_newAABBs);

#ifdef TEST_computeAABBs
        std::cout << std::format(
            "rootAABB computed by thrust::reduce is ({}, {}, {}) -> ({}, {}, {})\n",
            rootAABB.minPos.x, rootAABB.minPos.y, rootAABB.minPos.z, rootAABB.maxPos.x,
            rootAABB.maxPos.y, rootAABB.maxPos.z);
        AABB rootAABB = d_aabbs.front();
        std::cout << std::format("d_aabbs[0] is ({}, {}, {}) -> ({}, {}, {})\n", rootAABB.minPos.x,
                                 rootAABB.minPos.y, rootAABB.minPos.z, rootAABB.maxPos.x,
                                 rootAABB.maxPos.y, rootAABB.maxPos.z);
#endif // TEST_computeAABBs
    }

    isComplete = true;
}
