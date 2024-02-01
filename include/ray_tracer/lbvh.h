#ifndef KOUEK_RAY_TRACER_LBVH_H
#define KOUEK_RAY_TRACER_LBVH_H

#include <filesystem>
#include <string_view>

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

    struct KOUEK_CUDA_ALIGN InternalNode {
        // 0b1xx...xx: child is a leaf
        // 0b0xx...xx: child is an internal
        uint32_t lft;
        uint32_t rht;

        AABB aabb;

        static constexpr auto ChildIsLeafMask = uint32_t(0x80000000);
        static constexpr auto LeafIndexMask = uint32_t(0x7fffffff);
    };
    struct KOUEK_CUDA_ALIGN DeviceData {
        AABB rootAABB;
        IndexTy faceNum;
        IndexTy compactedFaceNum;
        IndexTy *faces;
        IndexTy *compactedFaces;
        InternalNode *internalNodes;
    };

  private:
    bool isComplete = false;

    thrust::device_vector<IndexTy> d_faces;
    thrust::device_vector<IndexTy> d_compactedFaces;
    thrust::device_vector<InternalNode> d_internalNodes;

    DeviceData deviceDat;
    DeviceData *d_deviceDatPtr = nullptr;

    static constexpr std::string_view StatTag = "[LBVH Status]";
    static constexpr std::string_view ErrTag = "[LBVH Error]";

  public:
    ~LBVH();

    struct InputMesh {
        const std::vector<glm::vec3> &positions;
        const std::vector<glm::vec<3, IndexTy>> &facePositionIndices;
    };
    void BuildFrom(const InputMesh &inputMesh, const std::filesystem::path &logPath = "");

    bool IsComplete() const { return isComplete; }

    const DeviceData *GetDeivceDataDevicePtr() const { return d_deviceDatPtr; }

#define CONST_REF_GETTER(member, memberNameInFunc)                                                 \
    const decltype(member) &Get##memberNameInFunc() const { return member; }
    CONST_REF_GETTER(deviceDat, DeviceData)
#undef CONST_REF_GETTER

  private:
    void clear();
    void uploadDeviceData();
};

} // namespace RayTracer
} // namespace kouek

#endif // !KOUEK_RAY_TRACER_LBVH_H
