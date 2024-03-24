#ifndef KOUEK_RAY_TRACER_RAY_TRACER_H
#define KOUEK_RAY_TRACER_RAY_TRACER_H

#include <memory>
#include <string_view>

#include <unordered_map>
#include <vector>

#include <glm/glm.hpp>

#include <curand_kernel.h>

#include <cuda/helper.h>
#include <cuda/texture.h>
#include <ray_tracer/lbvh.h>
#include <ray_tracer/material.h>
#include <util/meta_type.h>

namespace kouek {
namespace RayTracer {

class RayTracer : Noncopyable {
  public:
    using IndexTy = uint32_t;

    static constexpr float Eps = .001f;

    struct KOUEK_CUDA_ALIGN Triangle {
        glm::vec<3, IndexTy> normIdx;
        glm::vec<3, IndexTy> texCoordIdx;
        IndexTy grpIdx;
    };
    struct KOUEK_CUDA_ALIGN RenderParameter {
        uint8_t displayTreeHeight = 0;
        uint8_t maxPathDepth = 4;
        glm::mat4 w2s;
        glm::mat4 invProj;
        IndexTy lightNum = 0;

        Triangle *trianlges = nullptr;
        glm::vec<3, IndexTy> *triToPositionIndices = nullptr;
        glm::vec3 *positions = nullptr;
        glm::vec3 *normals = nullptr;
        glm::vec2 *texCoords = nullptr;
        float lightTotArea;
        Light *lights = nullptr;
        Material *materials = nullptr;
    };
    struct KOUEK_CUDA_ALIGN RenderParameterPerFrame {
        glm::vec3 eyePos2w;
        glm::mat3 eyeRot2w;
    };

    enum class RenderTarget {
        Scene,
        AABBs,
        Triangles,
        Lights,
        Positions,
        Normals,
        TextureCoords,
        None
    };

  private:
    float prevMaxSamplePerPixel = 1024.f;
    float maxSamplePerPixel = 1024.f;
    float currAccuSamplePerPixel = 0.f;

    cudaSurfaceObject_t prevRndrTo;
    glm::ivec2 prevRndrSz = glm::ivec2{0};
    RenderTarget prevRndrTarget = RenderTarget::None;

    Modifiable<RenderParameter> rndrParam;
    RenderParameter *d_rndrParamPtr = nullptr;

    Modifiable<RenderParameterPerFrame> rndrParamPerFrame;
    RenderParameterPerFrame *d_rndrParamPerFramePtr = nullptr;

    std::shared_ptr<LBVH> lbvh;

    static constexpr std::string_view ErrTag = "[RayTracer Error]";

  private:
    thrust::device_vector<Triangle> d_triangles;
    // Since visiting the index of vertex positions is frequently used in Ray-Obj intersection
    // testing, use a seperate SoA to store the mapping from faces to the vertex positions.
    thrust::device_vector<glm::vec<3, IndexTy>> d_triToPositionIndices;
    thrust::device_vector<glm::vec3> d_positions;
    thrust::device_vector<glm::vec3> d_normals;
    thrust::device_vector<glm::vec2> d_texCoords;
    thrust::device_vector<Light> d_lights;
    thrust::device_vector<Material> d_materials;

    // Since Surface Object storing uchar4 will lose precision,
    // use a memory storing float3 to keep average value
    thrust::device_vector<glm::vec3> d_accuRndrTo;
    thrust::device_vector<curandState> d_randStates;

  public:
    ~RayTracer();

    struct InputMesh {
        const std::vector<glm::vec3> &positions;
        const std::vector<glm::vec3> &normals;
        const std::vector<glm::vec2> &texCoords;
        const std::vector<IndexTy> &groupStartFaceIndices;
        const std::vector<glm::vec<3, IndexTy>> &facePositionIndices;
        const std::vector<glm::vec<3, IndexTy>> &faceNormalIndices;
        const std::vector<glm::vec<3, IndexTy>> &faceTexCoordIndices;
        const std::vector<Light> &lights;
        const std::vector<Material> &materials;
    };
    void SetMesh(const InputMesh &inputMesh);
    void SetLBVH(std::shared_ptr<LBVH> lbvh);
    void SetWorldToScene(const glm::mat4 &w2s);
    void SetCameraToWorld(const glm::vec3 &eyePos2w, const glm::mat3 &eyeRot2w);
    void SetProjection(const glm::mat4 &proj);
    void SetDisplayTreeHeight(uint8_t displayTreeHeight);
    void SetMaxPathDepth(uint8_t maxPathDepth);

    void Render(cudaSurfaceObject_t rndrTo, const glm::ivec2 &rndrSz,
                RenderTarget rndrTarget = RenderTarget::Scene);

#define VAL_SETTER_GETTER(member, memberNameInFunc)                                                \
    void Set##memberNameInFunc(const decltype(member) &##member) { this->member = ##member; }      \
    const decltype(member) Get##memberNameInFunc() const { return member; }
    VAL_SETTER_GETTER(maxSamplePerPixel, MaxSamplePerPixel)
#undef VAL_SETTER_GETTER

#define VAL_GETTER(member, memberNameInFunc)                                                       \
    const decltype(member) Get##memberNameInFunc() const { return member; }
    VAL_GETTER(currAccuSamplePerPixel, CurrentAccumulatedSamplePerPixel)
#undef VAL_GETTER
};

} // namespace RayTracer
} // namespace kouek

#endif // !KOUEK_RAY_TRACER_RAY_TRACER_H
