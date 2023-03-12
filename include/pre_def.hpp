#ifndef KOUEK_PRE_DEF_H
#define KOUEK_PRE_DEF_H

#include <functional>
#include <limits>

#include <cg/mesh.hpp>

#include <glm/glm.hpp>

#include <cuda.h>

namespace kouek {

template <typename T> struct Updatable {
    bool needUpdate = true;
    T val;

    const T &operator()() const { return val; }
    template <typename MemT> const MemT &operator()(MemT T::*memPtr) const {
        return val.*memPtr;
    }

    template <typename OtherTy> Updatable<T> &operator=(OtherTy &&other) {
        val = std::forward<OtherTy>(other);
        needUpdate = true;
        return *this;
    }
    template <typename MemT> MemT &operator[](MemT T::*memPtr) {
        needUpdate = true;
        return val.*memPtr;
    }

    template <typename CallTy> void Update(CallTy &&call) {
        if (needUpdate) {
            call();
            needUpdate = false;
        }
    }
};

struct RenderInfo {
    glm::uvec2 res;
    glm::uvec2 lowRes;
    glm::uint maxDepth;
};

struct AABBBox {
    glm::vec3 min;
    glm::vec3 max;

    // CUDA device code doesn't support struct with member that is NOT POD.
    // Thus, default constructor is not available here. Use a factory method
    // instead.
    static __host__ __device__ AABBBox Create() {
        AABBBox ret;
        ret.Reset();
        return ret;
    }
    inline __host__ __device__ void Reset() {
        for (uint8_t xyz = 0; xyz < 3; ++xyz) {
            min[xyz] = std::numeric_limits<float>::max();
            max[xyz] = std::numeric_limits<float>::min();
        }
    }
    inline __host__ __device__ void Expand(const glm::vec3 &pos) {
        for (uint8_t xyz = 0; xyz < 3; ++xyz) {
            if (min[xyz] > pos[xyz])
                min[xyz] = pos[xyz];
            if (max[xyz] < pos[xyz])
                max[xyz] = pos[xyz];
        }
    }
    inline __host__ __device__ void Expand(const AABBBox &other) {
        for (uint8_t xyz = 0; xyz < 3; ++xyz) {
            if (min[xyz] > other.min[xyz])
                min[xyz] = other.min[xyz];
            if (max[xyz] < other.max[xyz])
                max[xyz] = other.max[xyz];
        }
    }
    inline __host__ __device__ bool Contain(const AABBBox &other) const {
        for (uint8_t xyz = 0; xyz < 3; ++xyz) {
            if (min[xyz] > other.min[xyz])
                return false;
            if (max[xyz] < other.max[xyz])
                return false;
        }
        return true;
    }
    inline __host__ __device__ auto GetCenter() const {
        return (min + max) * .5f;
    }
    inline __host__ __device__ uint8_t GetMaxDim() const {
        auto rng = max - min;
        if (rng[0] >= rng[1] && rng[0] >= rng[2])
            return 0;
        if (rng[1] >= rng[0] && rng[1] >= rng[2])
            return 1;
        if (rng[2] >= rng[0] && rng[2] >= rng[1])
            return 2;
        return 0;
    }
    inline __host__ __device__ auto GetSurfArea() const {
        auto range = max - min;
        return 2.f *
               (range.x * range.y + range.y * range.z + range.y * range.z);
    }
};

struct BVHNode {
    static constexpr glm::uint BVHNodeLeafBitFlag =
        0b1 << (sizeof(glm::uint) * 8 - 1);
    enum class Type { NonLeaf, Leaf };

    Type type;
    // 1. If node.type is Leaf, dat[0] is the start face idx, dat[1] is
    // number of faces the node covers.
    // 2. If node.type is NonLeaf, dat[0] and dat[1] are the left and right
    // child node idx respectively.
    glm::uvec2 dat;
    AABBBox aabb;
};

struct SceneInfo {
    glm::uint faceNum;
    glm::uint lightNum;
    glm::vec3 bkgrndCol;
    const BVHNode *bvh;
    const glm::uint *grp2faceBVHNodeIndices;
    const glm::uint *grp2mtls;
    const glm::uint *groups;
    const glm::vec3 *positions;
    const glm::vec2 *uvs;
    const glm::vec3 *normals;
    const Mesh::Material *mtls;
    const Mesh::Light *lights;
    const Mesh::Face2Idx3 *faces;
};

// CUDA dim3 doesn't support constexpr construction, use const
const dim3 DefBlockDim1D{16 * 16};
const dim3 DefBlockDim2D{16, 16};

constexpr auto NoneIdx = std::numeric_limits<glm::uint>::max();
constexpr auto Eps = std::numeric_limits<float>::epsilon();

} // namespace kouek

#endif // !KOUEK_DEF_H
