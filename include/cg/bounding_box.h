#ifndef KOUEK_BOUNDING_BOX_H
#define KOUEK_BOUNDING_BOX_H

#include <glm/glm.hpp>

#include <cuda/helper.h>

namespace kouek {

struct AABB {
    glm::vec3 minPos;
    glm::vec3 maxPos;

    __host_dev__ void Init() {
        minPos.x = minPos.y = minPos.z = CUDA::FloatMax;
        maxPos.x = maxPos.y = maxPos.z = CUDA::FloatMin;
    }

    __host_dev__ void Init(const glm::vec3 &minPos, const glm::vec3 &maxPos) {
        this->minPos = minPos;
        this->maxPos = maxPos;
    }

    __host_dev__ void Expand(const glm::vec3 &pos) {
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
        for (uint8_t xyz = 0; xyz < 3; ++xyz) {
            if (minPos[xyz] > pos[xyz])
                minPos[xyz] = pos[xyz];
            if (maxPos[xyz] < pos[xyz])
                maxPos[xyz] = pos[xyz];
        }
    }

    __host_dev__ void Expand(const AABB &other) {
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
        for (uint8_t xyz = 0; xyz < 3; ++xyz) {
            if (minPos[xyz] > other.minPos[xyz])
                minPos[xyz] = other.minPos[xyz];
            if (maxPos[xyz] < other.maxPos[xyz])
                maxPos[xyz] = other.maxPos[xyz];
        }
    }

    __host_dev__ bool operator==(const AABB &other) const {
        return minPos == other.minPos && maxPos == other.maxPos;
    }
    __host_dev__ bool operator!=(const AABB &other) const { return !(*this == other); }

    __host_dev__ static constexpr AABB CreateInitial() {
        AABB ret;
        ret.minPos.x = ret.minPos.y = ret.minPos.z = CUDA::FloatMax;
        ret.maxPos.x = ret.maxPos.y = ret.maxPos.z = CUDA::FloatMin;
        return ret;
    }

    __host_dev__ static constexpr AABB CreateNormalized() {
        AABB ret;
        ret.minPos.x = ret.minPos.y = ret.minPos.z = 0.f;
        ret.maxPos.x = ret.maxPos.y = ret.maxPos.z = 1.f;
        return ret;
    }
};

} // namespace kouek

#endif // !KOUEK_BOUNDING_BOX_H
