#ifndef KOUEK_RAY_TRACER_LBVH_H
#define KOUEK_RAY_TRACER_LBVH_H

#include <string>

#include <array>
#include <vector>

#include <thrust/device_vector.h>

#include <glm/glm.hpp>

#include <cg/bounding_box.h>
#include <util/meta_type.h>

namespace kouek {
namespace RayTracer {

class LBVH : Noncopyable {
  public:
    using IndexTy = uint32_t;

    struct InternalNode {
        // 0b1xx...xx: child is a leaf
        // 0b0xx...xx: child is an internal
        uint32_t lft;
        uint32_t rht;

        static constexpr auto ChildIsLeafMask = uint32_t(0x80000000);
        static constexpr auto LeafIndexMask = uint32_t(0x7fffffff);
    };
    struct DeviceData {
        IndexTy faceNum;
        IndexTy compactedFaceNum;
        IndexTy *faces;
        IndexTy *compactedFaces;
    };

  private:
    AABB rootAABB;

    thrust::device_vector<IndexTy> d_faces;
    thrust::device_vector<IndexTy> d_compactedFaces;
    thrust::device_vector<AABB> d_aabbs;
    thrust::device_vector<InternalNode> d_internalNodes;

    std::string errMsg;

  public:
    struct InputMesh {
        const std::vector<glm::vec3> &positions;
        const std::vector<glm::vec<3, IndexTy>> &facePositionIndices;
    };
    void BuildFrom(const InputMesh &mesh);

    bool IsBuilded() const { return errMsg.empty(); }

#define CONST_REF_GETTER(member, memberNameInFunc)                                                 \
    const decltype(member) &Get##memberNameInFunc() const { return member; }
    CONST_REF_GETTER(rootAABB, RootAABB)
    CONST_REF_GETTER(errMsg, ErrorMessage)
#undef CONST_REF_GETTER
};

} // namespace RayTracer
} // namespace kouek

#endif // !KOUEK_RAY_TRACER_LBVH_H
