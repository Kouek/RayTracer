#ifndef KOUEK_RAY_TRACER_RAY_TRACER_H
#define KOUEK_RAY_TRACER_RAY_TRACER_H

#include <memory>
#include <string_view>

#include <unordered_map>
#include <vector>

#include <glm/glm.hpp>

#include <cuda/helper.h>
#include <ray_tracer/lbvh.h>
#include <util/meta_type.h>

namespace kouek {
namespace RayTracer {

class RayTracer : Noncopyable {
  public:
    using IndexTy = uint32_t;

    struct KOUEK_CUDA_ALIGN Triangle {
        glm::vec<3, IndexTy> normIdx;
        glm::vec<3, IndexTy> texCoordIdx;
        IndexTy grpIdx;
    };
    struct KOUEK_CUDA_ALIGN TriangleAttr {
        glm::vec3 norm;
        glm::vec2 texCoord;
    };
    struct KOUEK_CUDA_ALIGN RenderParameter {
        uint8_t displayHeight = 0;
        glm::mat4 w2s;
        glm::mat4 invProj;
        IndexTy triangleNum = 0;

        Triangle *trianlges = nullptr;
        glm::vec<3, IndexTy> *triToPositionIndices = nullptr;
        glm::vec3 *positions = nullptr;
        TriangleAttr *triAttrs = nullptr;
    };
    struct KOUEK_CUDA_ALIGN RenderParameterPerFrame {
        glm::vec3 eyePos2w;
        glm::mat3 eyeRot2w;
    };

  private:
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
    thrust::device_vector<TriangleAttr> d_triAttrs;

  public:
    ~RayTracer();

    struct InputMesh {
        const std::vector<glm::vec3> &positions;
        const std::vector<glm::vec3> &normals;
        const std::vector<glm::vec2> &texCoords;
        const std::vector<IndexTy> &groups;
        const std::vector<glm::vec<3, IndexTy>> &facePositionIndices;
        const std::vector<glm::vec<3, IndexTy>> &faceNormalIndices;
        const std::vector<glm::vec<3, IndexTy>> &faceTexCoordIndices;
        const std::unordered_map<IndexTy, std::string> &grp2mtls;
    };
    void SetMesh(const InputMesh &inputMesh);
    void SetLBVH(std::shared_ptr<LBVH> lbvh);
    void SetWorldToScene(const glm::mat4 &w2s);
    void SetCameraToWorld(const glm::vec3 &eyePos2w, const glm::mat3 &eyeRot2w);
    void SetProjection(const glm::mat4 &proj);
    void SetDisplayHeight(uint8_t displayHeight);

    enum class RenderTarget { Scene, AABBs, Triangles };
    void Render(cudaSurfaceObject_t rndrTo, const glm::ivec2 &rndrSz,
                RenderTarget rndrTarget = RenderTarget::Scene);
};

} // namespace RayTracer
} // namespace kouek

#endif // !KOUEK_RAY_TRACER_RAY_TRACER_H
