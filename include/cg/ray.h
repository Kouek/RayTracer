#ifndef KOUEK_RAY_H
#define KOUEK_RAY_H

#include <glm/glm.hpp>

#include <cg/bounding_box.h>
#include <cuda/helper.h>

namespace kouek {

struct Ray {
    glm::vec3 pos;
    glm::vec3 dir;

    struct HitResult {
        float tEnter;
        float tExit;
    };

    __host_dev__ HitResult Hit(const AABB &aabb) const {
        float ht[8];
        ht[0] = (aabb.minPos.x - pos.x) / dir.x;
        ht[1] = (aabb.maxPos.x - pos.x) / dir.x;
        ht[2] = (aabb.minPos.y - pos.y) / dir.y;
        ht[3] = (aabb.maxPos.y - pos.y) / dir.y;
        ht[4] = (aabb.minPos.z - pos.z) / dir.z;
        ht[5] = (aabb.maxPos.z - pos.z) / dir.z;
        ht[6] = glm::max(glm::max(glm::min(ht[0], ht[1]), glm::min(ht[2], ht[3])),
                         glm::min(ht[4], ht[5]));
        ht[7] = glm::min(glm::min(glm::max(ht[0], ht[1]), glm::max(ht[2], ht[3])),
                         glm::max(ht[4], ht[5]));
        ht[6] = (ht[6] < 0.f) ? 0.f : ht[6];
        return {ht[6], ht[7]};
    }
};

} // namespace kouek

#endif // !KOUEK_RAY_H
