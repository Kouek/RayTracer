#ifndef KOUEK_RAY_H
#define KOUEK_RAY_H

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include <cg/bounding_box.h>
#include <cuda/helper.h>

namespace kouek {

struct Ray {
    glm::vec3 pos;
    glm::vec3 dir;

    struct KOUEK_CUDA_ALIGN HitShellResult {
        float tEnter;
        float tExit;
    };
    struct KOUEK_CUDA_ALIGN HitTriangleResult {
        float t;
        glm::vec2 bary;
    };
    struct KOUEK_CUDA_ALIGN HitQuadResult {
        float t;
        float u, v;
    };

    KOUEK_CUDA_HOST_DEV HitShellResult Hit(const AABB &aabb) const {
        HitShellResult ret;

        float ht[6];
        ht[0] = (aabb.minPos.x - pos.x) / dir.x;
        ht[1] = (aabb.maxPos.x - pos.x) / dir.x;
        ht[2] = (aabb.minPos.y - pos.y) / dir.y;
        ht[3] = (aabb.maxPos.y - pos.y) / dir.y;
        ht[4] = (aabb.minPos.z - pos.z) / dir.z;
        ht[5] = (aabb.maxPos.z - pos.z) / dir.z;
        ret.tEnter = glm::max(glm::max(glm::min(ht[0], ht[1]), glm::min(ht[2], ht[3])),
                              glm::min(ht[4], ht[5]));
        ret.tExit = glm::min(glm::min(glm::max(ht[0], ht[1]), glm::max(ht[2], ht[3])),
                             glm::max(ht[4], ht[5]));
        ret.tEnter = (ret.tEnter < 0.f) ? 0.f : ret.tEnter;
        return ret;
    }

    struct KOUEK_CUDA_ALIGN HitTriangle {
        const glm::vec3 &p0;
        const glm::vec3 &p1;
        const glm::vec3 &p2;
    };
    KOUEK_CUDA_HOST_DEV HitTriangleResult Hit(HitTriangle tri) const {
        HitTriangleResult ret;
        if (glm::intersectRayTriangle(pos, dir, tri.p0, tri.p1, tri.p2, ret.bary, ret.t))
            return ret;
        ret.t = -1.f;
        return ret;
    }

    struct KOUEK_CUDA_ALIGN HitQuad {
        const glm::vec3 &o;
        const glm::vec3 &u;
        const glm::vec3 &v;
    };
    KOUEK_CUDA_HOST_DEV HitQuadResult Hit(HitQuad quad) const {
        HitQuadResult ret;
        if (!glm::intersectRayPlane(pos, dir, quad.o, glm::normalize(glm::cross(quad.u, quad.v)),
                                    ret.t)) {
            ret.t = -1.f;
            return ret;
        }

        auto xLocal = pos + ret.t * dir - quad.o;
        auto lenU = glm::length(quad.u);
        auto lenV = glm::length(quad.v);
        ret.u = glm::dot(xLocal, quad.u / lenU);
        ret.v = glm::dot(xLocal, quad.v / lenV);
        if (ret.u < 0.f || ret.u > lenU || ret.v < 0.f || ret.v > lenV) {
            ret.t = -1.f;
            return ret;
        }

        return ret;
    }

    struct KOUEK_CUDA_ALIGN HitSphere {
        float r;
        const glm::vec3 &o;
    };
    KOUEK_CUDA_HOST_DEV HitShellResult Hit(HitSphere sphere) const {
        // TODO
        return {0.f, 0.f};
    }
};

} // namespace kouek

#endif // !KOUEK_RAY_H
