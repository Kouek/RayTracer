#include <ray_caster/ray_caster.h>

#include <cg/math.h>

kouek::RayCaster::RayCaster::~RayCaster() {
    auto releaseWhenAllocated = [&](auto **d_datPtr) {
        using namespace kouek::CUDA;
        if (*d_datPtr) {
            KOUEK_CUDA_CHECK(cudaFree(*d_datPtr));
            *d_datPtr = nullptr;
        }
    };
    releaseWhenAllocated(&d_rndrParamPtr);
    releaseWhenAllocated(&d_rndrParamPerFramePtr);
}

void kouek::RayCaster::RayCaster::SetMaximumAlpha(float maxAlpha) {
    rndrParam.Set(&RenderParameter::maxAlpha, maxAlpha);
}

void kouek::RayCaster::RayCaster::SetSamplingRate(float samplingRate) {
    rndrParam.Set(&RenderParameter::samplingRate, samplingRate);
}

void kouek::RayCaster::RayCaster::SetTransferFunctionTexture(std::shared_ptr<CUDA::Texture> tfTex) {
    this->tfTex = tfTex;
}

void kouek::RayCaster::RayCaster::SetWorldToScene(const glm::mat4 &w2s) {
    rndrParam.Set(&RenderParameter::w2s, w2s);
}

void kouek::RayCaster::RayCaster::SetCameraToWorld(const glm::vec3 &eyePos2w,
                                                   const glm::mat3 &eyeRot2w) {
    rndrParamPerFrame.Set(&RenderParameterPerFrame::eyePos2w, eyePos2w);
    rndrParamPerFrame.Set(&RenderParameterPerFrame::eyeRot2w, eyeRot2w);
}

void kouek::RayCaster::RayCaster::SetProjection(const glm::mat4 &proj) {
    auto invProj = Math::InverseProjective(proj);
    rndrParam.Set(&RenderParameter::invProj, invProj);
}

void kouek::RayCaster::RayCaster::SetShading(const InputShadingParameter &inputShadingParam) {
    rndrParam.Set(&RenderParameter::useShading, true);
    rndrParam.Set(&RenderParameter::ka, inputShadingParam.ka);
    rndrParam.Set(&RenderParameter::kd, inputShadingParam.kd);
    rndrParam.Set(&RenderParameter::ks, inputShadingParam.ks);
    rndrParam.Set(&RenderParameter::shiness, inputShadingParam.shiness);
    rndrParam.Set(&RenderParameter::lightPosInVolumeSpace, inputShadingParam.lightPosInVolumeSpace);
    rndrParam.Set(&RenderParameter::lightCol, inputShadingParam.lightCol);
}

void kouek::RayCaster::RayCaster::UnsetShading() {
    rndrParam.Set(&RenderParameter::useShading, false);
}
