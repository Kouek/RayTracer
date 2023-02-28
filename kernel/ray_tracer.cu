#include "ray_tracer.cuh"

#include <device_launch_parameters.h>

#include <cg/math.hpp>
#include <cuda/cuda_helper.hpp>

using namespace kouek;

__constant__ RenderInfo dc_rndrInfo;
__constant__ SceneInfo dc_scnInfo;
__constant__ glm::mat3 dc_camRot;
__constant__ glm::vec3 dc_camPos;
__constant__ glm::mat4 dc_unProj;

void kouek::uploadRenderInfo(const RenderInfo &val) {
    CHECK_CUDA(cudaMemcpyToSymbol(dc_rndrInfo, &val, sizeof(val)));
}

void kouek::uploadSceneInfo(const SceneInfo &val) {
    CHECK_CUDA(cudaMemcpyToSymbol(dc_scnInfo, &val, sizeof(val)));
}

void kouek::uploadCamera(const glm::mat4 &val) {
    glm::mat3 rot = val;
    glm::vec3 pos = val[3];
    CHECK_CUDA(cudaMemcpyToSymbol(dc_camRot, &rot, sizeof(rot)));
    CHECK_CUDA(cudaMemcpyToSymbol(dc_camPos, &pos, sizeof(pos)));
}

void kouek::uploadProjection(const glm::mat4 &val) {
    auto unProj = Math::InverseProjective(val);
    CHECK_CUDA(cudaMemcpyToSymbol(dc_unProj, &unProj, sizeof(unProj)));
}

inline __device__ uchar4 rgbaFloat4ToUChar4(float4 f4) {
    f4.x = __saturatef(f4.x); // clamp to [0, 1]
    f4.y = __saturatef(f4.y);
    f4.z = __saturatef(f4.z);
    f4.w = __saturatef(f4.w);
    f4.x *= 255.f;
    f4.y *= 255.f;
    f4.z *= 255.f;
    f4.w *= 255.f;
    return make_uchar4(f4.x, f4.y, f4.z, f4.w);
}

inline __device__ bool rayIntersectAABB(float &tMinOut, float &tMaxOut,
                                        const glm::vec3 &rayPos,
                                        const glm::vec3 &rayDir,
                                        const glm::vec3 &minPos,
                                        const glm::vec3 &maxPos) {
    register float ts[8];
    ts[0] = (minPos.x - rayPos.x) / rayDir.x;
    ts[1] = (maxPos.x - rayPos.x) / rayDir.x;
    ts[2] = (minPos.y - rayPos.y) / rayDir.y;
    ts[3] = (maxPos.y - rayPos.y) / rayDir.y;
    ts[4] = (minPos.z - rayPos.z) / rayDir.z;
    ts[5] = (maxPos.z - rayPos.z) / rayDir.z;
    ts[6] =
        fmax(fmax(fmin(ts[0], ts[1]), fmin(ts[2], ts[3])), fmin(ts[4], ts[5]));
    ts[7] =
        fmin(fmin(fmax(ts[0], ts[1]), fmax(ts[2], ts[3])), fmax(ts[4], ts[5]));
    tMinOut = ts[6] < 0.f ? 0.f : ts[6];
    tMaxOut = ts[7];

    return ts[6] <= ts[7];
}

inline __device__ bool rayIntersectTriangle(float &tOut, float &uOut,
                                            float &vOut,
                                            const glm::vec3 &rayPos,
                                            const glm::vec3 &rayDir,
                                            glm::uint faceIdx) {
    auto &pi3 = dc_scnInfo.faces[faceIdx].v;
    glm::vec3 p3[3]{dc_scnInfo.positions[pi3[0]], dc_scnInfo.positions[pi3[1]],
                    dc_scnInfo.positions[pi3[2]]};
    register auto E1 = p3[1] - p3[0];
    register auto E2 = p3[2] - p3[0];
    register auto S = rayPos - p3[0];

    auto S1 = glm::cross(rayDir, E2);
    auto S2 = glm::cross(S, E1);
    auto inv = 1.f / glm::dot(S1, E1);
    auto t = inv * glm::dot(S2, E2);
    auto u = inv * glm::dot(S1, S);
    auto v = inv * glm::dot(S2, rayDir);
    if (t >= 0.f && u >= 0.f && v >= 0.f && (u + v) < 1.f) {
        tOut = t;
        uOut = u;
        vOut = v;
        return true;
    }
    return false;
}

__device__ bool rayIntersectBVH(glm::uint &outHitFaceIdx, glm::vec3 &outRayPos,
                                const glm::vec3 &rayDir) {
    struct StkElemTy {
        uint8_t visitCnt;
        glm::uint nodeIdx;
    };
    using StkPtrTy = uint8_t;
    static constexpr StkPtrTy MaxStkNum = 64;
    StkElemTy stk[MaxStkNum];
    StkPtrTy stkEndPtr = 0;

    auto push = [&](glm::uint nodeIdx) {
        if (stkEndPtr == MaxStkNum)
            return;
        auto &elem = stk[stkEndPtr++];
        elem.visitCnt = 0;
        elem.nodeIdx = nodeIdx;
    };
    auto pop = [&]() { --stkEndPtr; };
    auto top = [&]() -> StkElemTy & { return stk[stkEndPtr - 1]; };
    auto empty = [&]() { return stkEndPtr == 0; };

    float tNear = +INFINITY;
    push(0);
    while (!empty()) {
        auto &stkElem = top();
        if (stkElem.visitCnt == 2) {
            pop();
            continue;
        }

        const auto &curr = dc_scnInfo.bvh[stkElem.nodeIdx];

        float tMin, tMax;
        if (!rayIntersectAABB(tMin, tMax, outRayPos, rayDir, curr.aabb.min,
                              curr.aabb.max)) {
            pop();
            continue;
        }
        if (tMin > tNear) {
            pop();
            continue;
        }

        auto leafIsBVHNode = (curr.dat[0] & BVHNode::BVHNodeLeafBitFlag) != 0;

        if (curr.type == BVHNode::Type::NonLeaf) {
            auto childIdx = 0 == stkElem.visitCnt++ ? curr.dat[0] : curr.dat[1];
            if (leafIsBVHNode)
                childIdx ^= BVHNode::BVHNodeLeafBitFlag;
            push(childIdx);
            continue;
        }

        pop();

        auto idx = leafIsBVHNode ? curr.dat[0] ^ BVHNode::BVHNodeLeafBitFlag
                                 : curr.dat[0];
        for (glm::uint cnt = 0; cnt < curr.dat[1]; ++cnt, ++idx)
            if (leafIsBVHNode) {
                auto grpIdx = dc_scnInfo.groups[idx];
                auto faceBVHNodeIdx = dc_scnInfo.grp2FaceBVHNodeIndices[grpIdx];
                push(faceBVHNodeIdx);
            } else {
                float u, v;
                if (rayIntersectTriangle(tMin, u, v, outRayPos, rayDir, idx)) {
                    if (tNear > tMin) {
                        tNear = tMin;
                        outHitFaceIdx = idx;
                    }
                }
            }
    }

    if (tNear == +INFINITY)
        return false;

    outRayPos = outRayPos + tNear * rayDir;
    return true;
}

__global__ void rayTrace(cudaSurfaceObject_t output) {
    glm::uvec2 rndrPos{blockIdx.x * blockDim.x + threadIdx.x,
                       blockIdx.y * blockDim.y + threadIdx.y};
    if (rndrPos.x >= dc_rndrInfo.res.x || rndrPos.y >= dc_rndrInfo.res.y)
        return;

    glm::vec3 rayDir;
    {
        glm::vec4 tmp{(float)(rndrPos.x << 1) / dc_rndrInfo.res.x - 1.f,
                      (float)(rndrPos.y << 1) / dc_rndrInfo.res.y - 1.f, 1.f,
                      1.f};
        tmp = dc_unProj * tmp;
        rayDir = tmp;
        rayDir = dc_camRot * glm::normalize(rayDir);
    }

    auto setToBkgrnd = [&]() {
        auto &bkCol = dc_scnInfo.bkgrndCol;
        auto col =
            rgbaFloat4ToUChar4(make_float4(bkCol.r, bkCol.g, bkCol.b, 1.f));
        surf2Dwrite(col, output, rndrPos.x * 4, rndrPos.y);
    };

    auto rayPos = dc_camPos;
    glm::uint hitFaceIdx = NoneIdx;

    if (!rayIntersectBVH(hitFaceIdx, rayPos, rayDir)) {
        setToBkgrnd();
        return;
    }

    auto rng = dc_scnInfo.bvh[0].aabb.max - dc_scnInfo.bvh[0].aabb.min;
    auto p = (rayPos - dc_scnInfo.bvh[0].aabb.min) / rng;
    surf2Dwrite(rgbaFloat4ToUChar4(make_float4(p.x, p.y, p.z, 1.f)), output,
                rndrPos.x * 4, rndrPos.y);
}

void kouek::render(cudaSurfaceObject_t output, const glm::uvec2 &rndrRes) {
    dim3 gridDim{(rndrRes.x + DefBlockDim2D.x - 1) / DefBlockDim2D.x,
                 (rndrRes.y + DefBlockDim2D.y - 1) / DefBlockDim2D.y};
    rayTrace<<<gridDim, DefBlockDim2D>>>(output);
}
