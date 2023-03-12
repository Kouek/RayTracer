#include "ray_tracer.cuh"

#include <curand_kernel.h>
#include <device_launch_parameters.h>

#include <cg/math.hpp>
#include <cuda/cuda_helper.hpp>

#include "common.cuh"
#include "sample.cuh"

using namespace kouek;

__constant__ RenderInfo dc_rndrInfo;
__constant__ SceneInfo dc_scnInfo;
__constant__ glm::mat3 dc_camRot;
__constant__ glm::vec3 dc_camPos;
__constant__ glm::mat4 dc_unProj;

static curandState *d_curandStatePtr;
__constant__ curandState *dc_curandStatePtr;

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

__device__ bool rayHitAnyFace(const glm::vec3 &rayPos, const glm::vec3 &rayDir,
                              float maxDist) {
    struct StkElemTy {
        uint8_t visitCnt;
        glm::uint bvhIdx;
        glm::uint grpIdx;
    };
    using StkPtrTy = uint8_t;
    static constexpr StkPtrTy MaxStkNum = 64;
    StkElemTy stk[MaxStkNum];
    StkPtrTy stkEndPtr = 0;

    auto push = [&](glm::uint bvhIdx, glm::uint grpIdx = NoneIdx) {
        if (stkEndPtr == MaxStkNum)
            return;
        auto &elem = stk[stkEndPtr++];
        elem.visitCnt = 0;
        elem.bvhIdx = bvhIdx;
        elem.grpIdx = grpIdx;
    };
    auto pop = [&]() { --stkEndPtr; };
    auto top = [&]() -> StkElemTy & { return stk[stkEndPtr - 1]; };
    auto empty = [&]() { return stkEndPtr == 0; };

    push(0);
    while (!empty()) {
        auto &stkElem = top();
        auto gi = stkElem.grpIdx;
        if (stkElem.visitCnt == 2) {
            pop();
            continue;
        }

        const auto &curr = dc_scnInfo.bvh[stkElem.bvhIdx];

        float tMin, tMax;
        if (!rayIntersectAABB(tMin, tMax, rayPos, rayDir, curr.aabb.min,
                              curr.aabb.max)) {
            pop();
            continue;
        }
        if (tMin > maxDist) {
            pop();
            continue;
        }

        auto leafIsBVHNode = (curr.dat[0] & BVHNode::BVHNodeLeafBitFlag) != 0;

        if (curr.type == BVHNode::Type::NonLeaf) {
            auto childIdx = 0 == stkElem.visitCnt++ ? curr.dat[0] : curr.dat[1];
            if (leafIsBVHNode)
                childIdx ^= BVHNode::BVHNodeLeafBitFlag;
            push(childIdx, gi);
            continue;
        }

        pop();

        auto idx = leafIsBVHNode ? curr.dat[0] ^ BVHNode::BVHNodeLeafBitFlag
                                 : curr.dat[0];
        for (glm::uint cnt = 0; cnt < curr.dat[1]; ++cnt, ++idx)
            if (leafIsBVHNode) {
                auto grpIdx = dc_scnInfo.groups[idx];
                auto faceBVHNodeIdx = dc_scnInfo.grp2faceBVHNodeIndices[grpIdx];
                push(faceBVHNodeIdx, grpIdx);
            } else {
                auto &pi3 = dc_scnInfo.faces[idx].v;
                glm::vec3 p3[3]{dc_scnInfo.positions[pi3[0]],
                                dc_scnInfo.positions[pi3[1]],
                                dc_scnInfo.positions[pi3[2]]};
                glm::vec3 bary;
                if (rayIntersectTriangle(tMin, bary[1], bary[2], bary[0],
                                         rayPos, rayDir, p3))
                    if (tMin <= maxDist)
                        return true;
            }
    }

    return false;
}

__device__ bool rayHitCloseFace(State &outState, const glm::vec3 &rayPos,
                                const glm::vec3 &rayDir) {
    outState.t = +INFINITY;

    // Try intersecting with lights first,
    // because some lights may be insight the walls.
    for (glm::uint li = 0; li < dc_scnInfo.lightNum; ++li) {
        auto light = dc_scnInfo.lights[li];
        if (light.type == Mesh::Light::Type::Quad) {
            auto N =
                glm::normalize(glm::cross(light.quad.uv[0], light.quad.uv[1]));
            auto t = rayIntersectRect(rayPos, rayDir, light.quad.pos, N,
                                      light.quad.uv[0], light.quad.uv[1]);

            if (outState.t > t) {
                outState.t = t;
                outState.li = li;
            }
        } else {
            auto t = rayIntersectSphere(rayPos, rayDir, light.sphere.pos,
                                        light.sphere.radius);

            if (outState.t > t) {
                outState.t = t;
                outState.li = li;
            }
        }
    }
    if (outState.li != NoneIdx) {
        outState.fhp = rayPos + outState.t * rayDir;
        return true;
    }

    struct StkElemTy {
        uint8_t visitCnt;
        glm::uint bvhIdx;
        glm::uint grpIdx;
    };
    using StkPtrTy = uint8_t;
    static constexpr StkPtrTy MaxStkNum = 64;
    StkElemTy stk[MaxStkNum];
    StkPtrTy stkEndPtr = 0;

    auto push = [&](glm::uint bvhIdx, glm::uint grpIdx = NoneIdx) {
        if (stkEndPtr == MaxStkNum)
            return;
        auto &elem = stk[stkEndPtr++];
        elem.visitCnt = 0;
        elem.bvhIdx = bvhIdx;
        elem.grpIdx = grpIdx;
    };
    auto pop = [&]() { --stkEndPtr; };
    auto top = [&]() -> StkElemTy & { return stk[stkEndPtr - 1]; };
    auto empty = [&]() { return stkEndPtr == 0; };

    push(0);
    while (!empty()) {
        auto &stkElem = top();
        auto gi = stkElem.grpIdx;
        if (stkElem.visitCnt == 2) {
            pop();
            continue;
        }

        const auto &curr = dc_scnInfo.bvh[stkElem.bvhIdx];

        float tMin, tMax;
        if (!rayIntersectAABB(tMin, tMax, rayPos, rayDir, curr.aabb.min,
                              curr.aabb.max)) {
            pop();
            continue;
        }
        if (tMin > outState.t) {
            pop();
            continue;
        }

        auto leafIsBVHNode = (curr.dat[0] & BVHNode::BVHNodeLeafBitFlag) != 0;

        if (curr.type == BVHNode::Type::NonLeaf) {
            auto childIdx = 0 == stkElem.visitCnt++ ? curr.dat[0] : curr.dat[1];
            if (leafIsBVHNode)
                childIdx ^= BVHNode::BVHNodeLeafBitFlag;
            push(childIdx, gi);
            continue;
        }

        pop();

        auto idx = leafIsBVHNode ? curr.dat[0] ^ BVHNode::BVHNodeLeafBitFlag
                                 : curr.dat[0];
        for (glm::uint cnt = 0; cnt < curr.dat[1]; ++cnt, ++idx)
            if (leafIsBVHNode) {
                auto grpIdx = dc_scnInfo.groups[idx];
                auto faceBVHNodeIdx = dc_scnInfo.grp2faceBVHNodeIndices[grpIdx];
                push(faceBVHNodeIdx, grpIdx);
            } else {
                auto &pi3 = dc_scnInfo.faces[idx].v;
                glm::vec3 p3[3]{dc_scnInfo.positions[pi3[0]],
                                dc_scnInfo.positions[pi3[1]],
                                dc_scnInfo.positions[pi3[2]]};
                if (rayIntersectTriangle(tMin, outState.bary[1],
                                         outState.bary[2], outState.bary[0],
                                         rayPos, rayDir, p3)) {
                    if (outState.t > tMin) {
                        outState.t = tMin;
                        outState.gi = gi;
                        outState.fi = idx;
                    }
                }
            }
    }

    if (outState.t == +INFINITY)
        return false;

    outState.fhp = rayPos + outState.t * rayDir;
    return true;
}

__device__ glm::vec3 directLighting(const State &state, const glm::vec3 &rayDir,
                                    const Mesh::Material &mtl) {
    auto L = glm::zero<glm::vec3>();
    if (dc_scnInfo.lightNum == 0)
        return L;

    auto surfacePos =
        state.fhp + Eps * (state.isFront ? state.normal : -state.normal);

    auto light = [&]() {
        glm::uint li = curand_uniform(dc_curandStatePtr) * dc_scnInfo.lightNum;
        if (li == dc_scnInfo.lightNum)
            --li;
        return dc_scnInfo.lights[li];
    }();

    auto lightSample = lightSampling(light, dc_scnInfo.lightNum);
    auto lightDir = lightSample.surfacePos - surfacePos;
    auto lightDistSqr = lightDir.x * lightDir.x + lightDir.y * lightDir.y +
                        lightDir.z * lightDir.z;
    auto lightDist = glm::sqrt(lightDistSqr);
    lightDir /= lightDist;

    if (glm::dot(lightDir, state.isFront ? state.normal : -state.normal) <=
            .0f ||
        glm::dot(lightDir, lightSample.normal) >= .0f)
        return L;

    auto inShowdow = rayHitAnyFace(surfacePos, lightDir, lightDist - Eps);
    if (!inShowdow) {
        auto bsdfPDF = PhongPDF(state, mtl, rayDir, lightDir);
        auto f = PhongEval(state, mtl, rayDir, lightDir);

        auto lightPDF =
            lightDistSqr /
            (light.area * glm::abs(glm::dot(lightSample.normal, lightDir)));

        L += powerHeuristic(lightPDF, bsdfPDF) * f *
             glm::abs(glm::dot(state.isFront ? state.normal : -state.normal,
                               lightDir)) *
             lightSample.emission / lightPDF;
    }

    return L;
}

__global__ void rayTraceKernel(cudaSurfaceObject_t output, bool rndrLowRes,
                               bool reAccumulate) {
    glm::uvec2 rndrPos{blockIdx.x * blockDim.x + threadIdx.x,
                       blockIdx.y * blockDim.y + threadIdx.y};
    if (!rndrLowRes && rndrPos.x >= dc_rndrInfo.res.x ||
        rndrPos.y >= dc_rndrInfo.res.y)
        return;
    if (rndrLowRes && rndrPos.x >= dc_rndrInfo.lowRes.x &&
        rndrPos.y >= dc_rndrInfo.lowRes.y)
        return;

    glm::vec3 rayDir;
    {
        glm::vec4 tmp{
            (float)(rndrPos.x << 1) /
                    (rndrLowRes ? dc_rndrInfo.lowRes.x : dc_rndrInfo.res.x) -
                1.f,
            (float)(rndrPos.y << 1) /
                    (rndrLowRes ? dc_rndrInfo.lowRes.y : dc_rndrInfo.res.y) -
                1.f,
            1.f, 1.f};
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

    auto state = State::Create();
    auto rayPos = dc_camPos;

//#define BVH_TEST
    //#define NORMAL_TEST
    #define BARY_TEST
    //#define TEX_UV_TEST

#ifdef BVH_TEST

    if (!rayHitCloseFace(state, rayPos, rayDir)) {
        setToBkgrnd();
        return;
    }

    rayPos = (state.fhp - dc_scnInfo.bvh[0].aabb.min) /
             (dc_scnInfo.bvh[0].aabb.max - dc_scnInfo.bvh[0].aabb.min);
    auto col =
        rgbaFloat4ToUChar4(make_float4(rayPos.x, rayPos.y, rayPos.z, 1.f));
    surf2Dwrite(col, output, rndrPos.x * 4, rndrPos.y);

#elif defined(NORMAL_TEST)

    if (!rayHitCloseFace(state, rayPos, rayDir)) {
        setToBkgrnd();
        return;
    }

    auto idx3 = dc_scnInfo.faces[state.fi];
    state.normal = [&]() {
        auto &ni3 = idx3.n;
        return state.bary[0] * dc_scnInfo.normals[ni3[0]] +
               state.bary[1] * dc_scnInfo.normals[ni3[1]] +
               state.bary[2] * dc_scnInfo.normals[ni3[2]];
    }();
    state.isFront = glm::dot(rayDir, state.normal) <= 0.f;

    state.normal = .5f * (1.f + state.normal);
    auto col = rgbaFloat4ToUChar4(
        make_float4(state.normal.x, state.normal.y, state.normal.z, 1.f));
    if (state.isFront)
        surf2Dwrite(col, output, rndrPos.x * 4, rndrPos.y);
    else
        setToBkgrnd();

#elif defined(BARY_TEST)

    if (!rayHitCloseFace(state, rayPos, rayDir)) {
        setToBkgrnd();
        return;
    }

    if (state.li != NoneIdx) {
        surf2Dwrite(make_uchar4(255, 255, 255, 255), output, rndrPos.x * 4,
                    rndrPos.y);
        return;
    }

    auto idx3 = dc_scnInfo.faces[state.fi];
    auto col = rgbaFloat4ToUChar4(
        make_float4(state.bary.x, state.bary.y, state.bary.z, 1.f));
    surf2Dwrite(col, output, rndrPos.x * 4, rndrPos.y);

#elif defined(TEX_UV_TEST)

    if (!rayHitCloseFace(state, rayPos, rayDir)) {
        setToBkgrnd();
        return;
    }

    auto idx3 = dc_scnInfo.faces[state.fi];
    state.normal = [&]() {
        auto &ni3 = idx3.n;
        return state.bary[0] * dc_scnInfo.normals[ni3[0]] +
               state.bary[1] * dc_scnInfo.normals[ni3[1]] +
               state.bary[2] * dc_scnInfo.normals[ni3[2]];
    }();
    auto uv = [&]() {
        auto &ni3 = idx3.t;
        return state.bary[0] * dc_scnInfo.uvs[ni3[0]] +
               state.bary[1] * dc_scnInfo.uvs[ni3[1]] +
               state.bary[2] * dc_scnInfo.uvs[ni3[2]];
    }();
    state.isFront = glm::dot(rayDir, state.normal) <= 0.f;

    auto col = rgbaFloat4ToUChar4(make_float4(uv[0], uv[1], 0.f, 1.f));
    if (state.isFront)
        surf2Dwrite(col, output, rndrPos.x * 4, rndrPos.y);
    else
        setToBkgrnd();

#else

    auto accuColor = make_float4(0.f, 0.f, 0.f, 0.f);
    if (!reAccumulate) {
        uchar4 v4;
        surf2Dread(&v4, output, rndrPos.x * 4, rndrPos.y);
        accuColor = rgbaUChar4ToFloat4(v4);
    }

    glm::vec3 radiance{0.f};
    glm::vec3 throughout{1.f};
    auto lightSample = LightSample::Create();
    auto bsdfSample = BSDFSample::Create();
    for (; state.depth < dc_rndrInfo.maxDepth; ++state.depth) {
        if (!rayHitCloseFace(state, rayPos, rayDir)) {
            if (state.depth == 0)
                setToBkgrnd();
            return;
        }

        auto mtl = dc_scnInfo.mtls[dc_scnInfo.grp2mtls[state.gi]];
        auto idx3 = dc_scnInfo.faces[state.fi];
        state.normal = [&]() {
            auto &ni3 = idx3.n;
            return state.bary[0] * dc_scnInfo.normals[ni3[0]] +
                   state.bary[1] * dc_scnInfo.normals[ni3[1]] +
                   state.bary[2] * dc_scnInfo.normals[ni3[2]];
        }();
        state.isFront = glm::dot(rayDir, state.normal) <= 0.f;

        if (mtl.lightIdx != Mesh::NoneIdx) {
            auto light = dc_scnInfo.lights[mtl.lightIdx];
            lightSample.emission = light.radiance;

            if (light.type == Mesh::Light::Type::Quad) {
                auto cosTheta = glm::dot(-rayDir, state.normal);
                lightSample.pdf = (state.t * state.t) / (light.area * cosTheta);
            } else
                lightSample.pdf = (state.t * state.t) / light.area;

            radiance += emitterSampling(state, rayPos, rayDir, lightSample,
                                        bsdfSample) *
                        throughout;
            break;
        }

        if (mtl.tr.x == 1.f && mtl.tr.y == 1.f && mtl.tr.z == 1.f) {
            radiance += directLighting(state, rayDir, mtl) * throughout;

            bsdfSample.dir = PhongSampling(state, mtl, rayDir);
            bsdfSample.pdf = PhongPDF(state, mtl, rayDir, bsdfSample.dir);

            if (bsdfSample.pdf > 0.f)
                throughout *= PhongEval(state, mtl, rayDir, bsdfSample.dir) *
                              glm::abs(glm::dot(state.isFront ? state.normal
                                                              : -state.normal,
                                                bsdfSample.dir)) /
                              bsdfSample.pdf;
            else
                break;
        } else
            ;

        rayDir = bsdfSample.dir;
        rayPos = state.fhp + rayDir * Eps;
    }

    surf2Dwrite(rgbaFloat4ToUChar4(make_float4(accuColor.x + radiance.r,
                                               accuColor.y + radiance.g,
                                               accuColor.z + radiance.b, 1.f)),
                output, rndrPos.x * 4, rndrPos.y);
#endif // BVH_TEST
}

__global__ void setupCURANDStateKernel(curandState *outState,
                                       unsigned long long seed) {
    curand_init(seed, 0, 0, outState);
}

void kouek::render(cudaSurfaceObject_t output, const glm::uvec2 &rndrRes,
                   bool rndrLowRes, bool reAccumulate) {
    if (d_curandStatePtr == nullptr) {
        CHECK_CUDA(cudaMalloc(&d_curandStatePtr, sizeof(*d_curandStatePtr)));
        setupCURANDStateKernel<<<1, 1>>>(d_curandStatePtr, 0);
        CHECK_CUDA(cudaMemcpyToSymbol(dc_curandStatePtr, &d_curandStatePtr,
                                      sizeof(d_curandStatePtr)));
    }

    dim3 gridDim{(rndrRes.x + DefBlockDim2D.x - 1) / DefBlockDim2D.x,
                 (rndrRes.y + DefBlockDim2D.y - 1) / DefBlockDim2D.y};
    rayTraceKernel<<<gridDim, DefBlockDim2D>>>(output, rndrLowRes,
                                               reAccumulate);
}
