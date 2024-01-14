#include <ray_caster/ray_caster.h>

#include <format>
#include <type_traits>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include <cg/bounding_box.h>
#include <cg/math.h>
#include <cg/ray.h>

void kouek::RayCaster::RayCaster::SetRAWVolume(std::shared_ptr<CUDA::Texture> volTex) {
    using namespace kouek::CUDA;

    this->volTex = volTex;

    cudaResourceDesc resDesc;
    KOUEK_CUDA_CHECK(cudaGetTextureObjectResourceDesc(&resDesc, this->volTex->Get()));
    cudaChannelFormatDesc chnDesc;
    cudaExtent extent;
    KOUEK_CUDA_CHECK(cudaArrayGetInfo(&chnDesc, &extent, nullptr, resDesc.res.array.array));

    rawVolRndrParam.Set(&RAWVolumeRenderParameter::dim,
                        {extent.width, extent.height, extent.depth});
}

template <bool UseShading>
__device__ uchar4
renderScene(const kouek::RayCaster::RayCaster::RenderParameter &rndrParam,
            const kouek::RayCaster::RayCaster::RenderParameterPerFrame &rndrParamPerFrame,
            const kouek::RayCaster::RayCaster::RAWVolumeRenderParameter &rawVolRndrParam,
            cudaTextureObject_t volTex, cudaTextureObject_t tfTex, kouek::Ray &eyeRay) {
    using namespace kouek;

    auto hit = eyeRay.Hit(AABB::CreateNormalized());
    if (hit.tEnter >= hit.tExit)
        return uchar4{0, 0, 0, 0};

    uint32_t stepCnt = 0;
    uint32_t maxStepCnt = [&]() {
        auto dltPos = (hit.tExit - hit.tEnter) * eyeRay.dir;
        return glm::ceil(rndrParam.samplingRate * glm::length(dltPos) *
                         glm::max(rawVolRndrParam.dim.x,
                                  glm::max(rawVolRndrParam.dim.y, rawVolRndrParam.dim.z)));
    }();

    // Transform from scene to volume
    eyeRay.pos *= rawVolRndrParam.dim;
    eyeRay.dir = glm::normalize(eyeRay.dir * rawVolRndrParam.dim);
    hit = eyeRay.Hit(AABB{.minPos = {0.f, 0.f, 0.f}, .maxPos = rawVolRndrParam.dim});
    if (hit.tEnter >= hit.tExit)
        return uchar4{0, 0, 0, 0};

    auto step = (hit.tExit - hit.tEnter) / (maxStepCnt - 1);

    auto isPosInAABB = [&]() {
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
        for (uint8_t xyz = 0; xyz < 3; ++xyz)
            if (eyeRay.pos[xyz] < 0.f || eyeRay.pos[xyz] >= rawVolRndrParam.dim[xyz])
                return false;
        return true;
    };
    hit.tEnter = step * glm::ceil(hit.tEnter / step);
    eyeRay.pos += hit.tEnter * eyeRay.dir;
    if (!isPosInAABB())
        eyeRay.pos += step * eyeRay.dir;

    glm::vec3 rgb = {0.f, 0.f, 0.f};
    auto a = 0.f;

    while (stepCnt <= maxStepCnt && isPosInAABB()) {
        auto scalar = tex3D<float>(volTex, eyeRay.pos.x, eyeRay.pos.y, eyeRay.pos.z);
        auto tfCol = tex1D<float4>(tfTex, scalar);
        glm::vec3 tfColRGB = {tfCol.x, tfCol.y, tfCol.z};

        if constexpr (UseShading) {
            auto N = [&]() {
                glm::vec3 N;
                auto samplePos = eyeRay.pos;
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
                for (uint8_t xyz = 0; xyz < 3; ++xyz) {
                    samplePos[xyz] += .5f;
                    auto v0 = tex3D<float>(volTex, samplePos.x, samplePos.y, samplePos.z);
                    samplePos[xyz] -= 1.f;
                    auto v1 = tex3D<float>(volTex, samplePos.x, samplePos.y, samplePos.z);
                    N[xyz] = v1 - v0;
                }
                return glm::normalize(N);
            }();
            if (glm::dot(eyeRay.dir, N) > 0.f)
                N = -N;

            auto p2lDir = glm::normalize(rndrParam.lightPosInVolumeSpace - eyeRay.pos);
            auto ambient = rndrParam.ka * tfColRGB;
            auto diffuse =
                rndrParam.kd * glm::max(0.f, glm::dot(N, p2lDir)) * tfColRGB * rndrParam.lightCol;
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
            break;

        ++stepCnt;
        eyeRay.pos += step * eyeRay.dir;
    }

    kouek::Math::GammaCorrect(rgb);
    return uchar4{glm::clamp(static_cast<uint8_t>(255.f * rgb.r), uint8_t(0), uint8_t(255)),
                  glm::clamp(static_cast<uint8_t>(255.f * rgb.g), uint8_t(0), uint8_t(255)),
                  glm::clamp(static_cast<uint8_t>(255.f * rgb.b), uint8_t(0), uint8_t(255)),
                  glm::clamp(static_cast<uint8_t>(255.f * a), uint8_t(0), uint8_t(255))};
}

__device__ uchar4 renderAABB(kouek::Ray &eyeRay) {
    using namespace kouek;

    auto hit = eyeRay.Hit(AABB::CreateNormalized());
    if (hit.tEnter >= hit.tExit)
        return uchar4{0, 0, 0, 0};

    auto enterPos = eyeRay.pos + hit.tEnter * eyeRay.dir;
    auto exitPos = eyeRay.pos + hit.tExit * eyeRay.dir;

    auto color = .3f * enterPos + .7f * exitPos;
    return uchar4{glm::clamp(static_cast<uint8_t>(255.f * color.r), uint8_t(0), uint8_t(255)),
                  glm::clamp(static_cast<uint8_t>(255.f * color.g), uint8_t(0), uint8_t(255)),
                  glm::clamp(static_cast<uint8_t>(255.f * color.b), uint8_t(0), uint8_t(255)), 255};
}

void kouek::RayCaster::RayCaster::RenderRAWVolume(cudaSurfaceObject_t rndrTo,
                                                  const glm::ivec2 &rndrSz,
                                                  RAWVolumeRenderTarget rndrTarget) {
    if (!volTex || !tfTex) {
        std::cerr << std::format("{} at {}:{}. volTex or tfTex is not set.\n", ErrTag, __FILE__,
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
    uploadInNeed(rawVolRndrParam, &d_rawVolRndrParamPtr);

    auto flatRes = static_cast<glm::uint>(rndrSz.x) * rndrSz.y;
    thrust::for_each(
        thrust::make_counting_iterator(decltype(flatRes)(0)),
        thrust::make_counting_iterator(flatRes),
        [rndrTo, rndrSz, rndrTarget, volTex = volTex->Get(), tfTex = tfTex->Get(),
         rndrParamPtr = d_rndrParamPtr, rndrParamPerFramePtr = d_rndrParamPerFramePtr,
         rawVolRndrParamPtr = d_rawVolRndrParamPtr] __device__(decltype(flatRes) threadIdx) {
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
            case kouek::RayCaster::RayCaster::RAWVolumeRenderTarget::Scene:
                if (rndrParamPtr->useShading)
                    color = renderScene<true>(*rndrParamPtr, *rndrParamPerFramePtr,
                                              *rawVolRndrParamPtr, volTex, tfTex, eyeRay);
                else
                    color = renderScene<false>(*rndrParamPtr, *rndrParamPerFramePtr,
                                               *rawVolRndrParamPtr, volTex, tfTex, eyeRay);
                break;
            case kouek::RayCaster::RayCaster::RAWVolumeRenderTarget::AABB:
                color = renderAABB(eyeRay);
                break;
            }

            surf2Dwrite(color, rndrTo, sizeof(uchar4) * pix.x, pix.y);
        });
}
