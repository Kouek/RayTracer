#ifndef KOUEK_RAY_CASTER_TEST_OPTIONS_H
#define KOUEK_RAY_CASTER_TEST_OPTIONS_H

#include <format>
#include <iostream>

#include <glm/glm.hpp>

// #define TEST_computeNodesAndChildren
// #define TEST_rayCastVDB

#ifdef TEST_computeNodesAndChildren
#include <ray_caster/dpbx_vdb.h>

#include <queue>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <thrust/device_vector.h>

inline void test_computeNodesAndChildren(
    kouek::RayCaster::DepthBoxVDB::RelativeIndexTy leafNum,
    kouek::RayCaster::DepthBoxVDB::DeviceData deviceDat,
    const thrust::device_vector<kouek::RayCaster::DepthBoxVDB::Node> &d_nodePools,
    const thrust::device_vector<kouek::RayCaster::DepthBoxVDB::RelativeIndexTy> &d_childPools) {
    std::vector<kouek::RayCaster::DepthBoxVDB::Node> nodePools(d_nodePools.size());
    thrust::copy(d_nodePools.begin(), d_nodePools.end(), nodePools.begin());
    std::vector<kouek::RayCaster::DepthBoxVDB::RelativeIndexTy> childPools(d_childPools.size());
    thrust::copy(d_childPools.begin(), d_childPools.end(), childPools.begin());
    deviceDat.nodePools = nodePools.data();
    deviceDat.childPools = childPools.data();

    std::unordered_map<kouek::RayCaster::DepthBoxVDB::IndexTy,
                       kouek::RayCaster::DepthBoxVDB::IndexTy>
        inodeVisits;
    std::unordered_map<kouek::RayCaster::DepthBoxVDB::IndexTy,
                       kouek::RayCaster::DepthBoxVDB::IndexTy>
        leafVisits;
    std::queue<std::tuple<kouek::RayCaster::DepthBoxVDB::IndexTy, uint8_t>> que;
    que.emplace(deviceDat.vdbParam.nodeStarts[deviceDat.vdbParam.rootLev],
                deviceDat.vdbParam.rootLev);
    while (!que.empty()) {
        auto [idx, lev] = que.front();
        que.pop();

        auto childNum = deviceDat.vdbParam.dims[lev];
        childNum = childNum * childNum * childNum;

        auto &curr = deviceDat.nodePools[idx];
        auto childPtr = &(deviceDat.childPools[deviceDat.vdbParam.childStarts[lev - 1] +
                                               curr.childPoolOffsRelative]);
        for (kouek::RayCaster::DepthBoxVDB::RelativeIndexTy chi = 0; chi < childNum; ++chi) {
            auto idxRlt = childPtr[chi];
            if (idxRlt != kouek::RayCaster::DepthBoxVDB::UndefIdxU32)
                if (lev > 1) {
                    auto [itr, inserted] = inodeVisits.emplace(
                        std::make_pair(deviceDat.vdbParam.nodeStarts[lev - 1] + idxRlt, idx));
                    if (!inserted)
                        std::cout << std::format(
                            "[ERROR] inode {} on lev {} with parent {} has been visited\n",
                            deviceDat.vdbParam.nodeStarts[lev - 1] + idxRlt, lev - 1, idx);
                    else
                        que.emplace(itr->first, lev - 1);
                } else {
                    auto [itr, inserted] = leafVisits.emplace(
                        std::make_pair(deviceDat.vdbParam.nodeStarts[lev - 1] + idxRlt, idx));
                    if (!inserted)
                        std::cout << std::format(
                            "[ERROR] leaf {} with parent {} has been visited\n",
                            deviceDat.vdbParam.nodeStarts[lev - 1] + idxRlt, idx);
                }
        }
    }

    std::cout << std::format("{} == leafNum =?= leafVisits.size() == {}\n", leafNum,
                             leafVisits.size());
}
#endif // TEST_computeNodesAndChildren

#ifdef TEST_rayCastVDB
#include <ray_caster/dda.h>
#include <ray_caster/dpbx_vdb.h>

#include <array>
#include <vector>

inline void test_rayCastVDB(
    const kouek::RayCaster::DepthBoxVDB::DeviceData &_vdb,
    const thrust::device_vector<kouek::RayCaster::DepthBoxVDB::Node> &d_nodePools,
    const thrust::device_vector<kouek::RayCaster::DepthBoxVDB::RelativeIndexTy> &d_childPools,
    const thrust::device_vector<kouek::RayCaster::DepthBoxVDB::RelativeIndexTy>
        &d_atlasBrickToNodeIndices) {
    using namespace kouek;

    auto vdb = _vdb;
    std::vector<kouek::RayCaster::DepthBoxVDB::Node> nodePools(d_nodePools.size());
    std::vector<kouek::RayCaster::DepthBoxVDB::RelativeIndexTy> childPools(d_childPools.size());
    std::vector<kouek::RayCaster::DepthBoxVDB::RelativeIndexTy> atlasBrickToNodeIndices(
        d_atlasBrickToNodeIndices.size());
    thrust::copy(d_nodePools.begin(), d_nodePools.end(), nodePools.begin());
    thrust::copy(d_childPools.begin(), d_childPools.end(), childPools.begin());
    thrust::copy(d_atlasBrickToNodeIndices.begin(), d_atlasBrickToNodeIndices.end(),
                 atlasBrickToNodeIndices.begin());
    vdb.nodePools = nodePools.data();
    vdb.childPools = childPools.data();
    vdb.atlasBrickToNodeIndices = atlasBrickToNodeIndices.data();

    auto &vdbParam = vdb.vdbParam;

    std::array<Ray, 6 + 4> eyeRays;
    for (uint8_t i = 0; i < 6; ++i) {
        auto t0 = i / 2;
        auto t1 = i % 2;

        auto &eyeRay = eyeRays[i];
        eyeRay.dir.x = t0 != 0 ? 0.f : t1 == 0 ? 1.f : -1.f;
        eyeRay.dir.y = t0 != 1 ? 0.f : t1 == 0 ? 1.f : -1.f;
        eyeRay.dir.z = t0 != 2 ? 0.f : t1 == 0 ? 1.f : -1.f;
        eyeRay.pos = .5f * vdbParam.voxPerVDB - eyeRay.dir * vdbParam.voxPerVDB;
    }
    for (uint8_t i = 0; i < 4; ++i) {
        auto t0 = i & 0b01;
        auto t1 = i & 0b10;

        auto &eyeRay = eyeRays[6 + i];
        eyeRay.dir.x = t0 == 0 ? 1.f : -1.f;
        eyeRay.dir.y = t1 == 0 ? 1.f : -1.f;
        eyeRay.dir.z = 0.f;
        eyeRay.dir = glm::normalize(eyeRay.dir);
        eyeRay.pos = .5f * vdbParam.voxPerVDB - eyeRay.dir * vdbParam.voxPerVDB;
    }

    for (auto &eyeRay : eyeRays) {
        std::cout << std::format(
            "Ray from ({},{},{}) along ({},{},{}) starts ray-casting into VDB.\n", eyeRay.pos.x,
            eyeRay.pos.y, eyeRay.pos.z, eyeRay.dir.x, eyeRay.dir.y, eyeRay.dir.z);

        // Hit the whole VDB
        auto hit = eyeRay.Hit(
            AABB{.minPos = {0.f, 0.f, 0.f},
                 .maxPos = {vdbParam.voxPerVDB, vdbParam.voxPerVDB, vdbParam.voxPerVDB}});
        if (hit.tEnter >= hit.tExit) {
            std::cout << "Ray has no instersection with VDB.\n";
            return;
        }

        auto stk = RayCaster::VDBStack::Create(vdb);
        auto msgIndent = [&]() {
            for (uint8_t l = vdbParam.rootLev; l >= stk.lev; --l)
                std::cout << '\t';
        };
        auto msgPush = [&]() {
            msgIndent();
            std::cout << std::format("Push (lev,pos)=({},({},{},{})), tExit={} and span={}^3.\n",
                                     stk.lev, stk.TopNode().pos.x, stk.TopNode().pos.y,
                                     stk.TopNode().pos.z, stk.TopTExit(), vdbParam.dims[stk.lev]);
        };
        auto msgPop = [&]() {
            msgIndent();
            std::cout << std::format("Pop (lev,pos)=({},({},{},{})) and tExit={}.\n", stk.lev,
                                     stk.TopNode().pos.x, stk.TopNode().pos.y, stk.TopNode().pos.z,
                                     stk.TopTExit());
        };

        stk.Push(0, hit.tExit - RayCaster::Eps);
        msgPush();
        auto hdda3d = RayCaster::HDDA3D::Create(hit.tEnter + RayCaster::Eps, eyeRay);
        hdda3d.Prepare({0.f, 0.f, 0.f}, vdbParam.voxsPerChild[vdbParam.rootLev]);
        hdda3d.tCurr += RayCaster::Eps;

        while (!stk.Empty() && [&]() {
            for (uint8_t xyz = 0; xyz < 3; ++xyz)
                if (hdda3d.chPos[xyz] < 0 || hdda3d.chPos[xyz] >= vdbParam.dims[stk.lev])
                    return false;
            return true;
        }()) {
            hdda3d.Next();

            auto &curr = stk.TopNode();
            auto chIdxRlt = vdb.GetChild(stk.lev, curr.childPoolOffsRelative, hdda3d.chPos);

            if (chIdxRlt != RayCaster::DepthBoxVDB::UndefIdxU32) {
                msgIndent();
                std::cout << std::format("Child at ({},{},{}) is valid.\n", hdda3d.chPos.x,
                                         hdda3d.chPos.y, hdda3d.chPos.z);

                if (stk.lev == 1) {
                    auto leaf = vdb.GetNode(0, chIdxRlt);
                    hdda3d.tCurr += RayCaster::Eps;
                    auto posInBrick = eyeRay.pos + hdda3d.tCurr * eyeRay.dir - glm::vec3{leaf.pos};

                    msgIndent();
                    std::cout << std::format(
                        "Visit leaf ({},{},{}) with in-brick pos ({},{},{}) and atlas-brick-pos "
                        "({},{},{}).\n",
                        leaf.pos.x, leaf.pos.y, leaf.pos.z, posInBrick.x, posInBrick.y,
                        posInBrick.z, leaf.brickPosInAtlas.x, leaf.brickPosInAtlas.y,
                        leaf.brickPosInAtlas.z);

                    auto atlasBrickIdx = leaf.brickPosInAtlas.z * vdbParam.atlasBrickPerVDB.y *
                                             vdbParam.atlasBrickPerVDB.x +
                                         leaf.brickPosInAtlas.y * vdbParam.atlasBrickPerVDB.x +
                                         leaf.brickPosInAtlas.x;
                    auto nodeIdxRlt = vdb.atlasBrickToNodeIndices[atlasBrickIdx];
                    auto &node = vdb.GetNode(0, nodeIdxRlt);
                    msgIndent();
                    std::cout << std::format(
                        "Reversely, atlas-brick-to-node of ({},{},{}) is ({},{},{})\n",
                        leaf.brickPosInAtlas.x, leaf.brickPosInAtlas.y, leaf.brickPosInAtlas.z,
                        node.pos.x, node.pos.y, node.pos.z);
                    if (node.pos.x != leaf.pos.x || node.pos.y != leaf.pos.y ||
                        node.pos.z != leaf.pos.z) {
                        msgIndent();
                        std::cerr << "Wrong reverse mapping!!!\n";
                    }

                    hdda3d.Step();
                } else {
                    stk.Push(chIdxRlt, hdda3d.tNext - RayCaster::Eps);
                    msgPush();
                    hdda3d.tCurr += RayCaster::Eps;
                    hdda3d.Prepare(stk.TopNode().pos, vdbParam.voxsPerChild[stk.lev]);
                }
            } else {
                msgIndent();
                std::cout << std::format("Child at ({},{},{}) is empty.\n", hdda3d.chPos.x,
                                         hdda3d.chPos.y, hdda3d.chPos.z);
                hdda3d.Step();
            }

            while (hdda3d.tCurr >= stk.TopTExit()) {
                msgPop();
                stk.Pop();
                if (stk.Empty())
                    break;

                hdda3d.Prepare(stk.TopNode().pos, vdbParam.voxsPerChild[stk.lev]);
            }
        }
    }
}
#endif // TEST_rayCastVDB

#endif // !KOUEK_RAY_CASTER_TEST_OPTIONS_H
