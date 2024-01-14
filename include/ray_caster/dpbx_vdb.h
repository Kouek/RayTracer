#ifndef KOUEK_RAY_CASTER_DPBX_VDB_H
#define KOUEK_RAY_CASTER_DPBX_VDB_H

#include <format>
#include <limits>
#include <string_view>
#include <type_traits>

#include <vector>

#include <thrust/device_vector.h>

#include <glm/glm.hpp>

#include <cuda/texture.h>
#include <util/meta_type.h>

namespace kouek {
namespace RayCaster {

/* VDB Tree Structure Example :
 * If MaxLevelNum = 3 and rootLev = 2,
 * lev 2 (root): root -> childPool[chStarts[1]] of dims[2] ->
 * lev 1 (intr): nodePool[nodeStarts[1]] -> childPool[chStarts[0]] of dims[1] ->
 * lev 0 (leaf): nodePool[nodeStarts[0]] -> volume brick of dims[0]
 */
class DepthBoxVDB : Noncopyable {
  public:
    static constexpr auto MaxLevelNum = uint8_t(3);
    static constexpr auto MaxLog2Dim = uint8_t(10);

    using IndexTy = uint64_t;
    using RelativeIndexTy = uint32_t;
    using CoordValTy = int32_t;
    using CoordTy = glm::vec<3, CoordValTy>;

    static constexpr auto UndefRltIdx = std::numeric_limits<RelativeIndexTy>::max();

    struct KOUEK_CUDA_ALIGN VDBParameter {
        uint8_t rootLev;
        uint8_t apronWid;
        uint8_t apronDepWid;
        uint8_t log2Dims[MaxLevelNum];
        CoordValTy dims[MaxLevelNum];
        CoordValTy voxPerAtlasBrick;
        CoordValTy minDepPosValInBrick;
        CoordValTy maxDepPosValInBrick;
        glm::vec<3, CoordValTy> atlasBrickPerVDB;
        IndexTy nodeStarts[MaxLevelNum];
        IndexTy childStarts[MaxLevelNum];
        float voxsPerChild[MaxLevelNum];
        float voxPerVDB;
        glm::vec3 voxPerVol;
    };
    struct KOUEK_CUDA_ALIGN Node {
        CoordTy pos;
        CoordTy brickPosInAtlas;
        IndexTy childPoolOffsRelative;
    };
    struct KOUEK_CUDA_ALIGN DeviceData {
        Node *nodePools;
        RelativeIndexTy *childPools;
        RelativeIndexTy *atlasBrickToNodeIndices;
        cudaSurfaceObject_t atlasSurf;
        cudaTextureObject_t atlasTex;
        cudaTextureObject_t atlasDepTex;
        VDBParameter vdbParam;

        KOUEK_CUDA_HOST_DEV Node &GetRoot() const {
            return nodePools[vdbParam.nodeStarts[vdbParam.rootLev]];
        }
        KOUEK_CUDA_HOST_DEV Node &GetNode(uint8_t lev, RelativeIndexTy nodeIdxRelative) const {
            return nodePools[vdbParam.nodeStarts[lev] + nodeIdxRelative];
        }
        KOUEK_CUDA_HOST_DEV Node &GetNode(IndexTy nodeIdxAbsolute) const {
            return nodePools[nodeIdxAbsolute];
        }
        KOUEK_CUDA_HOST_DEV RelativeIndexTy GetChild(uint8_t parLev, RelativeIndexTy offsRelative,
                                                     const CoordTy &posRelative) const {
            auto idxRelative = PosRelativeToIdxRelative(parLev, posRelative);
            return childPools[vdbParam.childStarts[parLev - 1] + offsRelative + idxRelative];
        }
        KOUEK_CUDA_HOST_DEV RelativeIndexTy
        PosRelativeToIdxRelative(uint8_t parLev, const CoordTy &posRelative) const {
            auto log2Dim = vdbParam.log2Dims[parLev];
            return (posRelative.z << (log2Dim << 1)) | (posRelative.y << log2Dim) | posRelative.x;
        }
        KOUEK_CUDA_HOST_DEV RelativeIndexTy
        BrickPosInAtlasToNodeIdxRelative(const CoordTy &pos) const {
            return pos.z * vdbParam.atlasBrickPerVDB.y * vdbParam.atlasBrickPerVDB.x +
                   pos.y * vdbParam.atlasBrickPerVDB.x + pos.x;
        }
    };

  private:
    bool isComplete = false;

    thrust::device_vector<Node> d_nodePools;
    thrust::device_vector<RelativeIndexTy> d_childPools;
    thrust::device_vector<RelativeIndexTy> d_atlasBrickToNodeIndices;

    std::shared_ptr<kouek::CUDA::Array> atlasArr;
    std::unique_ptr<kouek::CUDA::Surface> atlasSurf;
    std::unique_ptr<kouek::CUDA::Texture> atlasTex;
    std::unique_ptr<kouek::CUDA::Texture> atlasDepTex;

    static constexpr std::string_view ErrTag = "[DepthBoxVDB Error]";

    DeviceData deviceDat;
    DeviceData *d_deviceDatPtr = nullptr;

  public:
    ~DepthBoxVDB();

    template <typename T> struct InputVolumeAndConfiguration {
        static_assert(std::is_same_v<T, uint8_t> || std::is_same_v<T, uint16_t>,
                      "T as voxel type is not implemented.");

        uint8_t apronWid = 1;
        std::array<uint8_t, MaxLevelNum> log2Dims = {4, 4, 3};
        bool useDepthBox = true;
        const kouek::CUDA::Texture &volTex;
        const std::vector<glm::vec<2, T>> &emptyScalarRngs;
    };
    template <typename T> void BuildFrom(const InputVolumeAndConfiguration<T> &inputVolAndCfg);

    bool IsComplete() const { return isComplete; }

    const DeviceData *GetDeivceDataDevicePtr() const { return d_deviceDatPtr; }

    size_t GetNodePoolsSize() const;
    size_t GetChildPoolsSize() const;
    size_t GetAtlasBrickToNodeIndicesSize() const;

#define CONST_REF_GETTER(member, memberNameInFunc)                                                 \
    const decltype(member) &Get##memberNameInFunc() const { return member; }
    CONST_REF_GETTER(deviceDat, DeviceData)
#undef CONST_REF_GETTER

  private:
    void clear();
    void uploadDeviceData();
};

} // namespace RayCaster
} // namespace kouek

#endif // !KOUEK_RAY_CASTER_DPBX_VDB_H
