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
#include <unordered_set>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

inline void test_computeInternalNodes(
    const thrust::device_vector<kouek::RayTracer::LBVH::InternalNode> &d_internalNodes,
    const thrust::device_vector<kouek::RayTracer::LBVH::IndexTy> &d_faces,
    const thrust::device_vector<kouek::RayTracer::LBVH::IndexTy> &d_compactedFaces) {
    using INodeTy = kouek::RayTracer::LBVH::InternalNode;
    using IndexTy = kouek::RayTracer::LBVH::IndexTy;

    thrust::host_vector<INodeTy> internalNodes(d_internalNodes);
    thrust::host_vector<IndexTy> faces(d_faces);
    thrust::host_vector<IndexTy> compactedFaces(d_compactedFaces);

    std::queue<std::tuple<IndexTy, IndexTy, IndexTy>> que;
    std::unordered_map<IndexTy, IndexTy> inodeVisits;
    std::unordered_map<IndexTy, IndexTy> leafVisits;
    std::unordered_set<IndexTy> faceVisits;
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
            else {
                auto fi = compactedFaces[leafIdx];
                auto fiEnd = leafIdx == compactedFaces.size() - 1 ? faces.size()
                                                                  : compactedFaces[leafIdx + 1];
                while (fi < fiEnd) {
                    auto face = faces[fi];
                    if (faceVisits.find(face) != faceVisits.end())
                        std::cout << std::format(
                            "[ERROR] face {} in compact {} with parent {} has been visited\n", face,
                            leafIdx, inodeIdx);
                    else
                        faceVisits.emplace(face);

                    ++fi;
                }
            }
        };

        auto &node = internalNodes[inodeIdx];
        auto testChild = [&](IndexTy chIdx) {
            if ((chIdx & INodeTy::ChildIsLeafMask) == 0)
                que.emplace(chIdx, inodeIdx, h + 1);
            else
                visitLeaf(chIdx & INodeTy::LeafIndexMask);
        };
        testChild(node.lft);
        testChild(node.rht);
    }

    std::cout << std::format("inodeVisits.size() == {} =?= {} == internalNodes.size()\n",
                             inodeVisits.size(), internalNodes.size());
    std::cout << std::format("leafVisits.size() == {} =?= {} == leafNum\n", leafVisits.size(),
                             compactedFaces.size());
    std::cout << std::format("faceVisits.size() == {} =?= {} == faceNum\n", faceVisits.size(),
                             faces.size());
    std::cout << std::format("lBVH has height {}\n", treeHeight);
}

#endif // TEST_computeInternalNodes

#endif // !KOUEK_RAY_TRACER_TEST_OPTIONS_H
