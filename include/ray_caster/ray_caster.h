#ifndef KOUEK_RAY_CASTER_RAY_CASTER_H
#define KOUEK_RAY_CASTER_RAY_CASTER_H

#include <memory>
#include <string_view>

#include <glm/glm.hpp>

#include <cuda/texture.h>
#include <ray_caster/dpbx_vdb.h>
#include <util/meta_type.h>

namespace kouek {
namespace RayCaster {

class RayCaster : Noncopyable {
  public:
    struct KOUEK_CUDA_ALIGN RenderParameter {
        bool useShading = false;
        glm::mat4 w2s;
        glm::mat4 invProj;
        glm::vec3 spaces;
        float maxAlpha = .9f;
        float samplingRate = 3.f;
        float ka, kd, ks, shiness;
        glm::vec3 lightPosInVolumeSpace;
        glm::vec3 lightCol;
    };
    struct KOUEK_CUDA_ALIGN RenderParameterPerFrame {
        glm::vec3 eyePos2w;
        glm::mat3 eyeRot2w;
    };
    struct KOUEK_CUDA_ALIGN RAWVolumeRenderParameter {
        glm::vec3 dim;
    };
    struct KOUEK_CUDA_ALIGN DPBXVDBRenderParameter {
        uint8_t displayLev;
        uint8_t displayLeafLayer;
    };

  private:
    Modifiable<RenderParameter> rndrParam;
    RenderParameter *d_rndrParamPtr = nullptr;

    Modifiable<RenderParameterPerFrame> rndrParamPerFrame;
    RenderParameterPerFrame *d_rndrParamPerFramePtr = nullptr;

    Modifiable<RAWVolumeRenderParameter> rawVolRndrParam;
    RAWVolumeRenderParameter *d_rawVolRndrParamPtr = nullptr;

    Modifiable<DPBXVDBRenderParameter> dpbxvdbRndrParam;
    DPBXVDBRenderParameter *d_dpbxvdbRndrParamPtr = nullptr;

    std::shared_ptr<CUDA::Texture> tfTex;
    std::shared_ptr<CUDA::Texture> volTex;
    std::shared_ptr<DepthBoxVDB> vdb;

    static constexpr std::string_view ErrTag = "[RayCaster Error]";

  public:
    ~RayCaster();

    void SetRAWVolume(std::shared_ptr<CUDA::Texture> volTex);
    void SetDepthBoxVDB(std::shared_ptr<DepthBoxVDB> vdb);
    void SetDepthBoxVDBDisplayLevel(uint8_t lev);
    void SetDepthBoxVDBDisplayLeafLayer(uint8_t layer);
    void SetMaximumAlpha(float maxAlpha);
    void SetSamplingRate(float samplingRate);
    void SetTransferFunctionTexture(std::shared_ptr<CUDA::Texture> tfTex);
    void SetWorldToScene(const glm::mat4 &w2s);
    void SetCameraToWorld(const glm::vec3 &eyePos2w, const glm::mat3 &eyeRot2w);
    void SetProjection(const glm::mat4 &proj);

    struct InputShadingParameter {
        float ka, kd, ks, shiness;
        glm::vec3 lightPosInVolumeSpace;
        glm::vec3 lightCol;
    };
    void SetShading(const InputShadingParameter &inputShadingParam);
    void UnsetShading();

    enum class RAWVolumeRenderTarget { Scene, AABB };
    void RenderRAWVolume(cudaSurfaceObject_t rndrTo, const glm::ivec2 &rndrSz,
                         RAWVolumeRenderTarget rndrTarget = RAWVolumeRenderTarget::Scene);
    enum class DPBXVDBRenderTarget { Scene, AABBs, Depths, PositionsAfterDepthSkip };
    template <typename T>
    void RenderDepthBoxVDB(cudaSurfaceObject_t rndrTo, const glm::ivec2 &rndrSz,
                           DPBXVDBRenderTarget rndrTarget = DPBXVDBRenderTarget::Scene);
};

} // namespace RayCaster
} // namespace kouek

#endif // !KOUEK_RAY_CASTER_RAY_CASTER_H
