#include <ray_caster/dda.h>
#include <ray_caster/ray_caster.h>

#include <format>
#include <iostream>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include <cg/bounding_box.h>
#include <cg/math.h>
#include <cg/ray.h>

void kouek::RayCaster::RayCaster::SetDepthBoxVDB(std::shared_ptr<DepthBoxVDB> vdb) {
    this->vdb = vdb;
}

void kouek::RayCaster::RayCaster::SetDepthBoxVDBDisplayLevel(uint8_t lev) {
    dpbxvdbRndrParam.Set(&DPBXVDBRenderParameter::displayLev, lev);
}

void kouek::RayCaster::RayCaster::SetDepthBoxVDBDisplayLeafLayer(uint8_t layer) {
    dpbxvdbRndrParam.Set(&DPBXVDBRenderParameter::displayLeafLayer, layer);
}

struct InLeafFuncParameter {
    float tCurr;
    float tExit;
    kouek::RayCaster::DepthBoxVDB::Node leaf;
};

template <typename SceneSpaceCallbackTy, typename VolumeSpaceCallbackTy,
          typename VDBSpaceCallbackTy, typename InLeafCallbackTy, typename OnPushedChildCallbackTy,
          typename OnSteppedCallbackTy>
struct RayCastCallbacks {
    SceneSpaceCallbackTy scnSpaceCallback;
    VolumeSpaceCallbackTy volSpaceCallback;
    VDBSpaceCallbackTy vdbSpaceCallback;
    OnPushedChildCallbackTy onPushedChildCallback;
    OnSteppedCallbackTy onSteppedCallback;
    InLeafCallbackTy inLeafCallback;
};

template <bool CullWhenNotHitVol = true, typename RayCastCallbacksTy>
__device__ void rayCastVDB(const kouek::RayCaster::RayCaster::RenderParameter &rndrParam,
                           const kouek::RayCaster::DepthBoxVDB::DeviceData &vdb, kouek::Ray &eyeRay,
                           RayCastCallbacksTy callbacks) {
    using namespace kouek;

    auto &vdbParam = vdb.vdbParam;

    // Hit the scene
    auto hit = eyeRay.Hit(AABB::CreateNormalized());
    if constexpr (CullWhenNotHitVol)
        if (hit.tEnter >= hit.tExit)
            return;
    if constexpr (!std::is_same_v<decltype(callbacks.scnSpaceCallback), nullptr_t>)
        callbacks.scnSpaceCallback(hit.tEnter, hit.tExit);

    // Transform from scene to VDB
    eyeRay.pos *= vdbParam.voxPerVol;
    eyeRay.dir = glm::normalize(eyeRay.dir * vdbParam.voxPerVol);

    // Hit the volume
    hit = eyeRay.Hit(AABB{.minPos = {0.f, 0.f, 0.f}, .maxPos = vdbParam.voxPerVol});
    if constexpr (CullWhenNotHitVol)
        if (hit.tEnter >= hit.tExit)
            return;
    if constexpr (!std::is_same_v<decltype(callbacks.volSpaceCallback), nullptr_t>)
        callbacks.volSpaceCallback(hit.tEnter, hit.tExit);

    // Hit the VDB
    hit = eyeRay.Hit(AABB{.minPos = {0.f, 0.f, 0.f},
                          .maxPos = {vdbParam.voxPerVDB, vdbParam.voxPerVDB, vdbParam.voxPerVDB}});
    if constexpr (!std::is_same_v<decltype(callbacks.vdbSpaceCallback), nullptr_t>)
        callbacks.vdbSpaceCallback(hit.tEnter, hit.tExit);
    if (hit.tEnter >= hit.tExit)
        return;

    auto stk = RayCaster::VDBStack::Create(vdb);
    stk.Push(0, hit.tExit - RayCaster::Eps);
    auto hdda3d = RayCaster::HDDA3D::Create(hit.tEnter + RayCaster::Eps, eyeRay);
    hdda3d.Prepare({0.f, 0.f, 0.f}, vdbParam.voxsPerChild[vdbParam.rootLev]);
    if constexpr (!std::is_same_v<decltype(callbacks.onPushedChildCallback), nullptr_t>)
        callbacks.onPushedChildCallback(hdda3d.tCurr, hdda3d.tNext, stk.lev, stk.TopNode());

    while (!stk.Empty() && [&]() {
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
        for (uint8_t xyz = 0; xyz < 3; ++xyz)
            if (hdda3d.chPos[xyz] < 0 || hdda3d.chPos[xyz] >= vdbParam.dims[stk.lev])
                return false;
        return true;
    }()) {
        hdda3d.Next();

        auto &curr = stk.TopNode();
        auto chIdxRlt = vdb.GetChild(stk.lev, curr.childPoolOffsRelative, hdda3d.chPos);

        if (chIdxRlt != RayCaster::DepthBoxVDB::UndefRltIdx)
            if (stk.lev == 1) {
                hdda3d.tCurr += RayCaster::Eps;
                InLeafFuncParameter param{.tCurr = hdda3d.tCurr,
                                          .tExit = hdda3d.tNext - RayCaster::Eps};
                param.leaf = vdb.GetNode(0, chIdxRlt);

                if constexpr (!std::is_same_v<decltype(callbacks.inLeafCallback), nullptr_t>)
                    if (!callbacks.inLeafCallback(param))
                        break;

                hdda3d.Step();
                if constexpr (!std::is_same_v<decltype(callbacks.onSteppedCallback), nullptr_t>)
                    callbacks.onSteppedCallback(hdda3d.tCurr, hdda3d.tNext, stk.lev, stk.TopNode());
            } else {
                stk.Push(chIdxRlt, hdda3d.tNext - RayCaster::Eps);
                hdda3d.tCurr += RayCaster::Eps;
                hdda3d.Prepare(stk.TopNode().pos, vdbParam.voxsPerChild[stk.lev]);
                if constexpr (!std::is_same_v<decltype(callbacks.onPushedChildCallback), nullptr_t>)
                    callbacks.onPushedChildCallback(hdda3d.tCurr, hdda3d.tNext, stk.lev,
                                                    stk.TopNode());
            }
        else {
            hdda3d.Step();
            if constexpr (!std::is_same_v<decltype(callbacks.onSteppedCallback), nullptr_t>)
                callbacks.onSteppedCallback(hdda3d.tCurr, hdda3d.tNext, stk.lev, stk.TopNode());
        }

        while (hdda3d.tCurr >= stk.TopTExit()) {
            stk.Pop();
            if (stk.Empty())
                break;

            hdda3d.Prepare(stk.TopNode().pos, vdbParam.voxsPerChild[stk.lev]);
        }
    }
}

template <typename T>
__device__ bool depthSkip(InLeafFuncParameter &param, const glm::vec3 &posInBrick,
                          const glm::vec3 &atlasBrickMinVoxPos,
                          const kouek::RayCaster::DepthBoxVDB::DeviceData &vdb,
                          kouek::Ray &eyeRay) {
    kouek::RayCaster::DepthDDA2D ddda2d;
    if (!ddda2d.Init(param.tCurr, vdb.vdbParam.voxsPerChild[1] - 1.f,
                     vdb.vdbParam.minDepPosValInBrick, vdb.vdbParam.maxDepPosValInBrick, posInBrick,
                     eyeRay))
        return false;

    while (true) {
        auto dep = tex3D<T>(vdb.atlasDepTex, atlasBrickMinVoxPos.x + ddda2d.posInBrick.x,
                            atlasBrickMinVoxPos.y + ddda2d.posInBrick.y,
                            atlasBrickMinVoxPos.z + ddda2d.posInBrick.z);
        if (dep <= ddda2d.dep)
            break;
        if (ddda2d.t >= param.tExit)
            return true;

        param.tCurr = ddda2d.t;
        ddda2d.StepNext();
    }
    return false;
}

template <typename T, bool UseDepthBox, bool UseShading>
__device__ uchar4 renderScene(cudaTextureObject_t tfTex,
                              const kouek::RayCaster::RayCaster::RenderParameter &rndrParam,
                              const kouek::RayCaster::DepthBoxVDB::DeviceData &vdb,
                              kouek::Ray &eyeRay) {
    auto rgb = glm::vec3{0.f, 0.f, 0.f};
    auto a = 0.f;
    float step;
    uint32_t maxStepCnt;
    glm::vec3 dltPos;

    RayCastCallbacks callbacks = {
        [&](float tEnter, float tExit) {
            auto tDlt = tExit - tEnter;
            maxStepCnt = [&]() {
                auto dltPos = tDlt * eyeRay.dir;
                return glm::ceil(
                    rndrParam.samplingRate * glm::length(dltPos) *
                    glm::max(vdb.vdbParam.voxPerVol.x,
                             glm::max(vdb.vdbParam.voxPerVol.y, vdb.vdbParam.voxPerVol.z)));
            }();
        },
        [&](float tEnter, float tExit) {
            auto tDlt = tExit - tEnter;
            step = tDlt / (maxStepCnt - 1);
            dltPos = step * eyeRay.dir;
        },
        nullptr,
        nullptr,
        nullptr,
        [&](InLeafFuncParameter &param) {
            param.tCurr = step * glm::ceil(param.tCurr / step);
            auto brickMinVoxPos = glm::vec3{param.leaf.pos};
            auto posInBrick = eyeRay.pos + param.tCurr * eyeRay.dir - brickMinVoxPos;

            glm::vec3 atlasBrickMinVoxPos =
                param.leaf.brickPosInAtlas * vdb.vdbParam.voxPerAtlasBrick +
                static_cast<kouek::RayCaster::DepthBoxVDB::CoordValTy>(vdb.vdbParam.apronDepWid);

            if constexpr (UseDepthBox) {
                if (depthSkip<T>(param, posInBrick, atlasBrickMinVoxPos, vdb, eyeRay))
                    return true;

                param.tCurr = step * glm::ceil(param.tCurr / step);
                posInBrick = eyeRay.pos + param.tCurr * eyeRay.dir - brickMinVoxPos;
            }

            while (param.tCurr < param.tExit && [&]() {
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
                for (uint8_t xyz = 0; xyz < 3; ++xyz)
                    if (posInBrick[xyz] < 0.f || posInBrick[xyz] >= vdb.vdbParam.voxsPerChild[1])
                        return false;
                return true;
            }()) {
                auto samplePos = atlasBrickMinVoxPos + posInBrick;
                auto scalar = tex3D<float>(vdb.atlasTex, samplePos.x, samplePos.y, samplePos.z);
                auto tfCol = tex1D<float4>(tfTex, scalar);
                auto tfColRGB = glm::vec3{tfCol.x, tfCol.y, tfCol.z};

                if constexpr (UseShading) {
                    auto N = [&]() {
                        glm::vec3 N;
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
                        for (uint8_t xyz = 0; xyz < 3; ++xyz) {
                            samplePos[xyz] += .5f;
                            auto v0 =
                                tex3D<float>(vdb.atlasTex, samplePos.x, samplePos.y, samplePos.z);
                            samplePos[xyz] -= 1.f;
                            auto v1 =
                                tex3D<float>(vdb.atlasTex, samplePos.x, samplePos.y, samplePos.z);
                            N[xyz] = v1 - v0;
                        }
                        return glm::normalize(N);
                    }();
                    if (glm::dot(eyeRay.dir, N) > 0.f)
                        N = -N;

                    auto p2lDir = glm::normalize(rndrParam.lightPosInVolumeSpace - brickMinVoxPos -
                                                 posInBrick);
                    auto ambient = rndrParam.ka * tfColRGB;
                    auto diffuse = rndrParam.kd * glm::max(0.f, glm::dot(N, p2lDir)) * tfColRGB *
                                   rndrParam.lightCol;
                    auto specular = [&]() {
                        auto hfDir = glm::normalize(p2lDir - eyeRay.dir);
                        return rndrParam.ks *
                               glm::pow(glm::max(0.f, glm::dot(N, hfDir)), rndrParam.shiness) *
                               rndrParam.lightCol;
                    }();
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
                    for (uint8_t xyz = 0; xyz < 3; ++xyz)
                        tfColRGB[xyz] = ambient[xyz] + diffuse[xyz] + specular[xyz];
                }

                rgb = rgb + (1.f - a) * tfCol.w * tfColRGB;
                a = a + (1.f - a) * tfCol.w;
                if (a >= rndrParam.maxAlpha)
                    return false;

                param.tCurr += step;
                posInBrick += dltPos;
            }

            return true;
        }};
    rayCastVDB(rndrParam, vdb, eyeRay, callbacks);

    kouek::Math::GammaCorrect(rgb);
    return uchar4{glm::clamp(static_cast<uint8_t>(255.f * rgb.r), uint8_t(0), uint8_t(255)),
                  glm::clamp(static_cast<uint8_t>(255.f * rgb.g), uint8_t(0), uint8_t(255)),
                  glm::clamp(static_cast<uint8_t>(255.f * rgb.b), uint8_t(0), uint8_t(255)),
                  glm::clamp(static_cast<uint8_t>(255.f * a), uint8_t(0), uint8_t(255))};
}

__device__ uchar4 renderAABBs(uint8_t displayLev,
                              const kouek::RayCaster::RayCaster::RenderParameter &rndrParam,
                              const kouek::RayCaster::DepthBoxVDB::DeviceData &vdb,
                              kouek::Ray &eyeRay) {
    static constexpr auto AlphaPerLev = .5f;
    auto rgb = glm::vec3{0.f, 0.f, 0.f};
    auto a = 0.f;

    RayCastCallbacks callbacks = {
        nullptr,
        nullptr,
        nullptr,
        [&](float tCurr, float tNext, uint8_t lev,
            const kouek::RayCaster::DepthBoxVDB::Node &node) {
            if (lev == displayLev) {
                auto pos = (eyeRay.pos + tCurr * eyeRay.dir - glm::vec3{node.pos}) /
                           (lev == vdb.vdbParam.rootLev ? vdb.vdbParam.voxPerVDB
                                                        : vdb.vdbParam.voxsPerChild[lev + 1]);
                rgb = rgb + (1.f - a) * AlphaPerLev * pos;
                a = a + (1.f - a) * AlphaPerLev;
            }
        },
        nullptr,
        [&](const InLeafFuncParameter &param) {
            if (displayLev == 0) {
                auto posInBrick = eyeRay.pos + param.tCurr * eyeRay.dir - glm::vec3{param.leaf.pos};

                rgb = rgb + (1.f - a) * AlphaPerLev * (posInBrick / vdb.vdbParam.voxsPerChild[1]);
                a = a + (1.f - a) * AlphaPerLev;
            }

            return true;
        }};
    rayCastVDB<false>(rndrParam, vdb, eyeRay, callbacks);

    return uchar4{glm::clamp(static_cast<uint8_t>(255.f * rgb.r), uint8_t(0), uint8_t(255)),
                  glm::clamp(static_cast<uint8_t>(255.f * rgb.g), uint8_t(0), uint8_t(255)),
                  glm::clamp(static_cast<uint8_t>(255.f * rgb.b), uint8_t(0), uint8_t(255)),
                  glm::clamp(static_cast<uint8_t>(255.f * a), uint8_t(0), uint8_t(255))};
}

template <typename T>
__device__ uchar4 renderDepths(uint8_t displayLeafLayer,
                               const kouek::RayCaster::RayCaster::RenderParameter &rndrParam,
                               const kouek::RayCaster::DepthBoxVDB::DeviceData &vdb,
                               kouek::Ray &eyeRay) {
    auto rgb = glm::vec3{0.f, 0.f, 0.f};
    auto a = 0.f;
    decltype(displayLeafLayer) layer = 0;

    RayCastCallbacks callbacks = {
        nullptr, nullptr, nullptr, nullptr, nullptr, [&](const InLeafFuncParameter &param) {
            if (layer == displayLeafLayer) {
                auto posInBrick = eyeRay.pos + param.tCurr * eyeRay.dir - glm::vec3{param.leaf.pos};

                glm::vec3 atlasBrickMinVoxPos =
                    param.leaf.brickPosInAtlas * vdb.vdbParam.voxPerAtlasBrick +
                    static_cast<kouek::RayCaster::DepthBoxVDB::CoordValTy>(
                        vdb.vdbParam.apronDepWid);

                kouek::RayCaster::DepthDDA2D ddda2d;
                if (ddda2d.Init(param.tCurr, vdb.vdbParam.voxsPerChild[1] - 1.f,
                                vdb.vdbParam.minDepPosValInBrick, vdb.vdbParam.maxDepPosValInBrick,
                                posInBrick, eyeRay)) {
                    auto dep =
                        tex3D<T>(vdb.atlasDepTex, atlasBrickMinVoxPos.x + ddda2d.posInBrick.x,
                                 atlasBrickMinVoxPos.y + ddda2d.posInBrick.y,
                                 atlasBrickMinVoxPos.z + ddda2d.posInBrick.z);
                    auto depNorm = dep / vdb.vdbParam.voxsPerChild[1];

                    rgb.r = rgb.g = rgb.b = depNorm;
                    a = 1.f;

                    return false;
                }

                rgb.r = 1.f;
                rgb.g = rgb.b = 0.f;
                a = 1.f;
                return false;
            } else {
                ++layer;
                return true;
            }
        }};

    rayCastVDB(rndrParam, vdb, eyeRay, callbacks);

    return uchar4{glm::clamp(static_cast<uint8_t>(255.f * rgb.r), uint8_t(0), uint8_t(255)),
                  glm::clamp(static_cast<uint8_t>(255.f * rgb.g), uint8_t(0), uint8_t(255)),
                  glm::clamp(static_cast<uint8_t>(255.f * rgb.b), uint8_t(0), uint8_t(255)),
                  glm::clamp(static_cast<uint8_t>(255.f * a), uint8_t(0), uint8_t(255))};
}

template <typename T>
__device__ uchar4 renderPositionsAfterDepthSkip(
    uint8_t displayLeafLayer, const kouek::RayCaster::RayCaster::RenderParameter &rndrParam,
    const kouek::RayCaster::DepthBoxVDB::DeviceData &vdb, kouek::Ray &eyeRay) {
    auto rgb = glm::vec3{0.f, 0.f, 0.f};
    auto a = 0.f;
    float step;
    uint32_t maxStepCnt;
    decltype(displayLeafLayer) layer = 0;

    RayCastCallbacks callbacks = {
        [&](float tEnter, float tExit) {
            auto tDlt = tExit - tEnter;
            maxStepCnt = [&]() {
                auto dltPos = tDlt * eyeRay.dir;
                return glm::ceil(
                    rndrParam.samplingRate * glm::length(dltPos) *
                    glm::max(vdb.vdbParam.voxPerVol.x,
                             glm::max(vdb.vdbParam.voxPerVol.y, vdb.vdbParam.voxPerVol.z)));
            }();
        },
        [&](float tEnter, float tExit) {
            auto tDlt = tExit - tEnter;
            step = tDlt / (maxStepCnt - 1);
        },
        nullptr,
        nullptr,
        nullptr,
        [&](InLeafFuncParameter &param) {
            if (layer == displayLeafLayer) {
                param.tCurr = step * glm::ceil(param.tCurr / step);
                auto brickMinVoxPos = glm::vec3{param.leaf.pos};
                auto posInBrick = eyeRay.pos + param.tCurr * eyeRay.dir - brickMinVoxPos;

                glm::vec3 atlasBrickMinVoxPos =
                    param.leaf.brickPosInAtlas * vdb.vdbParam.voxPerAtlasBrick +
                    static_cast<kouek::RayCaster::DepthBoxVDB::CoordValTy>(
                        vdb.vdbParam.apronDepWid);

                if (depthSkip<T>(param, posInBrick, atlasBrickMinVoxPos, vdb, eyeRay))
                    return true;

                param.tCurr = step * glm::ceil(param.tCurr / step);
                posInBrick = eyeRay.pos + param.tCurr * eyeRay.dir - brickMinVoxPos;

                rgb = posInBrick / vdb.vdbParam.voxsPerChild[1];
                a = 1.f;
                return false;
            } else {
                ++layer;
                return true;
            }
        }};
    rayCastVDB(rndrParam, vdb, eyeRay, callbacks);

    return uchar4{glm::clamp(static_cast<uint8_t>(255.f * rgb.r), uint8_t(0), uint8_t(255)),
                  glm::clamp(static_cast<uint8_t>(255.f * rgb.g), uint8_t(0), uint8_t(255)),
                  glm::clamp(static_cast<uint8_t>(255.f * rgb.b), uint8_t(0), uint8_t(255)),
                  glm::clamp(static_cast<uint8_t>(255.f * a), uint8_t(0), uint8_t(255))};
}

template <typename T>
void kouek::RayCaster::RayCaster::RenderDepthBoxVDB(cudaSurfaceObject_t rndrTo,
                                                    const glm::ivec2 &rndrSz,
                                                    DPBXVDBRenderTarget rndrTarget) {
    if (!vdb || !tfTex) {
        std::cerr << std::format("{} at {}:{}. vdb or tfTex are not set.\n", ErrTag, __FILE__,
                                 __LINE__);
        return;
    }

    auto uploadInNeed = [](auto &modifiable, auto **d_datPtrPtr) {
        using namespace kouek::CUDA;
        if (modifiable.modified) {
            if (!(*d_datPtrPtr))
                KOUEK_CUDA_CHECK(cudaMalloc(d_datPtrPtr, sizeof(**d_datPtrPtr)));

            auto &dat = modifiable.GetAndReset();
            KOUEK_CUDA_CHECK(cudaMemcpy(*d_datPtrPtr, &dat,
                                        sizeof(std::remove_reference_t<decltype(dat)>),
                                        cudaMemcpyHostToDevice));
        }
    };
    uploadInNeed(rndrParam, &d_rndrParamPtr);
    uploadInNeed(rndrParamPerFrame, &d_rndrParamPerFramePtr);
    uploadInNeed(dpbxvdbRndrParam, &d_dpbxvdbRndrParamPtr);

    auto flatRes = static_cast<glm::uint>(rndrSz.x) * rndrSz.y;
    thrust::for_each(
        thrust::make_counting_iterator(decltype(flatRes)(0)),
        thrust::make_counting_iterator(flatRes),
        [rndrTo, rndrSz, rndrTarget, tfTex = tfTex->Get(), rndrParamPtr = d_rndrParamPtr,
         rndrParamPerFramePtr = d_rndrParamPerFramePtr, dpbxvdbRndrParamPtr = d_dpbxvdbRndrParamPtr,
         vdbPtr = vdb->GetDeivceDataDevicePtr()] __device__(decltype(flatRes) threadIdx) {
            glm::ivec2 pix;
            pix.y = threadIdx / rndrSz.x;
            pix.x = threadIdx - pix.y * rndrSz.x;

            Ray eyeRay;
            {
                // Map [0, rndrSz.xy - 1] to (-1, 1)
                glm::vec4 tmp;
                tmp.z = rndrSz.x;
                tmp.w = rndrSz.y;
                tmp.x = (2.f * pix.x + 1.f - tmp.z) / tmp.z;
                tmp.y = (2.f * pix.y + 1.f - tmp.w) / tmp.w;

                // Inverse-project
                tmp.z = 1.f;
                tmp.w = 1.f;
                tmp = rndrParamPtr->invProj * tmp;

                eyeRay.dir = tmp;
                eyeRay.dir = rndrParamPerFramePtr->eyeRot2w * glm::normalize(eyeRay.dir);

                // Transform from world to scene
                tmp = glm::vec4(eyeRay.dir, 0.f);
                tmp = rndrParamPtr->w2s * tmp;
                eyeRay.dir = glm::normalize(eyeRay.dir);

                tmp = glm::vec4(rndrParamPerFramePtr->eyePos2w, 1.f);
                tmp = rndrParamPtr->w2s * tmp;
                eyeRay.pos = tmp;
            }

            uchar4 color;
            switch (rndrTarget) {
            case DPBXVDBRenderTarget::Scene:
                if (vdbPtr->vdbParam.apronDepWid == vdbPtr->vdbParam.apronWid)
                    if (rndrParamPtr->useShading)
                        color = renderScene<T, false, true>(tfTex, *rndrParamPtr, *vdbPtr, eyeRay);
                    else
                        color = renderScene<T, false, false>(tfTex, *rndrParamPtr, *vdbPtr, eyeRay);
                else if (rndrParamPtr->useShading)
                    color = renderScene<T, true, true>(tfTex, *rndrParamPtr, *vdbPtr, eyeRay);
                else
                    color = renderScene<T, true, false>(tfTex, *rndrParamPtr, *vdbPtr, eyeRay);
                break;
            case DPBXVDBRenderTarget::AABBs:
                color =
                    renderAABBs(dpbxvdbRndrParamPtr->displayLev, *rndrParamPtr, *vdbPtr, eyeRay);
                break;
            case DPBXVDBRenderTarget::Depths:
                color = renderDepths<T>(dpbxvdbRndrParamPtr->displayLeafLayer, *rndrParamPtr,
                                        *vdbPtr, eyeRay);
                break;
            case kouek::RayCaster::RayCaster::DPBXVDBRenderTarget::PositionsAfterDepthSkip:
                color = renderPositionsAfterDepthSkip<T>(dpbxvdbRndrParamPtr->displayLeafLayer,
                                                         *rndrParamPtr, *vdbPtr, eyeRay);
                break;
            }

            surf2Dwrite(color, rndrTo, sizeof(uchar4) * pix.x, pix.y);
        });
}

template void kouek::RayCaster::RayCaster::RenderDepthBoxVDB<uint8_t>(cudaSurfaceObject_t,
                                                                      const glm::ivec2 &,
                                                                      DPBXVDBRenderTarget);
template void kouek::RayCaster::RayCaster::RenderDepthBoxVDB<uint16_t>(cudaSurfaceObject_t,
                                                                       const glm::ivec2 &,
                                                                       DPBXVDBRenderTarget);
