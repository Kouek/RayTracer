#ifndef KOUEK_RAY_TRACER_TEST_OPTIONS_H
#define KOUEK_RAY_TRACER_TEST_OPTIONS_H

#include <format>
#include <iostream>

#include <glm/glm.hpp>

// #define TEST_computeInternalNodes
// #define TEST_computeAABBs

#ifdef TEST_computeInternalNodes
#include <ray_tracer/lbvh.h>

#include <queue>
#include <tuple>
#include <unordered_map>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

inline void test_computeInternalNodes(
    const thrust::device_vector<kouek::RayTracer::LBVH::InternalNode> &d_internalNodes) {
    using INodeTy = kouek::RayTracer::LBVH::InternalNode;
    using IndexTy = kouek::RayTracer::LBVH::IndexTy;

    thrust::host_vector<INodeTy> internalNodes(d_internalNodes);

    std::queue<std::tuple<IndexTy, IndexTy, IndexTy>> que;
    std::unordered_map<IndexTy, IndexTy> inodeVisits;
    std::unordered_map<IndexTy, IndexTy> leafVisits;
    que.emplace(0, std::numeric_limits<IndexTy>::max(), 1);
    uint32_t treeHeight = 0;
    while (!que.empty()) {
        auto [inodeIdx, par, h] = que.front();
        que.pop();

        if (treeHeight < h)
            treeHeight = h;

        auto [itr, inserted] = inodeVisits.emplace(std::make_pair(inodeIdx, par));
        if (!inserted) {
            std::cout << std::format("[ERROR] inode {} with parent {} has been visited\n", inodeIdx,
                                     par);
            continue;
        }

        auto visitLeaf = [&](decltype(INodeTy::lft) leafIdx) {
            auto [itr, inserted] = leafVisits.emplace(std::make_pair(leafIdx, inodeIdx));
            if (!inserted)
                std::cout << std::format("[ERROR] leaf {} with parent {} has been visited\n",
                                         leafIdx, inodeIdx);
        };

        auto &node = internalNodes[inodeIdx];
        if ((node.lft & INodeTy::ChildIsLeafMask) == 0)
            que.emplace(node.lft, inodeIdx, h + 1);
        else
            visitLeaf(node.lft & INodeTy::LeafIndexMask);
        if ((node.rht & INodeTy::ChildIsLeafMask) == 0)
            que.emplace(node.rht, inodeIdx, h + 1);
        else
            visitLeaf(node.rht & INodeTy::LeafIndexMask);
    }

    std::cout << std::format("inodeVisits.size() == {} =?= {} == internalNodes.size()\n",
                             inodeVisits.size(), internalNodes.size());
    std::cout << std::format("leafVisits.size() == {} =?= {} == leafNum\n", leafVisits.size(),
                             internalNodes.size() + 1);
    std::cout << std::format("lBVH has height {}\n", treeHeight);
}

#endif // TEST_computeInternalNodes

#endif // !KOUEK_RAY_TRACER_TEST_OPTIONS_H
