#ifndef KOUEK_RAY_TRACE_SCN_H
#define KOUEK_RAY_TRACE_SCN_H

#include <memory>

#include <vector>

#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cg/mesh.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <pre_def.hpp>

namespace kouek {

class RayTraceScn {
  public:
    static constexpr glm::uint LeafMaxFaceNum = 8;

  private:
    glm::uint bvhHeight = 0;
    glm::vec3 bkgrndCol = glm::zero<glm::vec3>();

    std::shared_ptr<Mesh> mesh;

    thrust::device_vector<glm::vec3> d_positions;
    thrust::device_vector<glm::vec2> d_uvs;
    thrust::device_vector<glm::vec3> d_normals;
    thrust::device_vector<Mesh::Material> d_mtls;
    thrust::device_vector<Mesh::Face2Idx3> d_faces;
    thrust::device_vector<glm::uint> d_groups;
    thrust::device_vector<glm::uint> d_grp2mtls;

    thrust::device_vector<glm::uint> d_grp2faceBVHNodeIndices;
    thrust::device_vector<BVHNode> d_bvh;

  public:
    void SetModel(std::shared_ptr<Mesh> mesh);
    inline auto GetBackgroundColor() const { return bkgrndCol; }
    inline void SetBackgroundColor(const glm::vec3 &val) { bkgrndCol = val; }
    void BuildBVH();
    SceneInfo GetScnInfo() const;
};

} // namespace kouek

#endif // !KOUEK_RAY_TRACE_SCN_H
