#ifndef KOUEK_RAY_CASTER_DDA_H
#define KOUEK_RAY_CASTER_DDA_H

#include <cg/ray.h>

#include <ray_caster/dpbx_vdb.h>

namespace kouek {
namespace RayCaster {

constexpr auto Eps = .01f;

struct KOUEK_CUDA_ALIGN VDBStack {
    int8_t lev;
    float tExits[DepthBoxVDB::MaxLevelNum - 1];
    DepthBoxVDB::Node nodes[DepthBoxVDB::MaxLevelNum - 1];
    const DepthBoxVDB::DeviceData &vdb;

    KOUEK_CUDA_HOST_DEV static VDBStack Create(const DepthBoxVDB::DeviceData &vdb) {
        VDBStack stk = {.lev = vdb.vdbParam.rootLev + 1, .vdb = vdb};
        return stk;
    }
    KOUEK_CUDA_HOST_DEV void Push(DepthBoxVDB::RelativeIndexTy nodeIdxRelative, float tExit) {
        --lev;
        nodes[lev - 1] = vdb.GetNode(lev, nodeIdxRelative);
        tExits[lev - 1] = tExit;
    }
    KOUEK_CUDA_HOST_DEV DepthBoxVDB::Node &TopNode() { return nodes[lev - 1]; }
    KOUEK_CUDA_HOST_DEV float TopTExit() { return tExits[lev - 1]; }
    KOUEK_CUDA_HOST_DEV void Pop() { ++lev; }
    KOUEK_CUDA_HOST_DEV bool Empty() { return lev == vdb.vdbParam.rootLev + 1; }
    KOUEK_CUDA_HOST_DEV bool Full() { return lev == 0; }
};

struct KOUEK_CUDA_ALIGN HDDA3D {
    glm::vec<3, int32_t> sign;  // signs of ray dir
    glm::vec<3, int32_t> mask;  // 0 for should NOT and 1 for should move on XYZ axis
    glm::vec<3, int32_t> chPos; // pos of child relative to its parent node
    float tCurr, tNext;
    glm::vec3 tSide;      // time that ray intersects with next plane in XYZ direction
    glm::vec3 tDlt;       // time delta
    const glm::vec3 &pos; // ray pos in VDB Space
    const glm::vec3 &dir; // ray dir in VDB Spacek

    KOUEK_CUDA_HOST_DEV static HDDA3D Create(float tCurr, const kouek::Ray &ray) {
        HDDA3D hdda3d = {.pos = ray.pos, .dir = ray.dir};
        hdda3d.sign = {hdda3d.dir.x > 0.f   ? 1
                       : hdda3d.dir.x < 0.f ? -1
                                            : 0,
                       hdda3d.dir.y > 0.f   ? 1
                       : hdda3d.dir.y < 0.f ? -1
                                            : 0,
                       hdda3d.dir.z > 0.f   ? 1
                       : hdda3d.dir.z < 0.f ? -1
                                            : 0};
        hdda3d.tCurr = tCurr;

        return hdda3d;
    }

    KOUEK_CUDA_HOST_DEV void Prepare(const glm::vec3 &min, float voxPerChild) {
        tDlt = glm::abs(voxPerChild / dir);
        auto pFlt = (pos + tCurr * dir - min) / voxPerChild;
        tSide = ((glm::floor(pFlt) - pFlt + .5f) * glm::vec3{sign} + .5f) * tDlt + tCurr;
        chPos = glm::floor(pFlt);
    }

    KOUEK_CUDA_HOST_DEV void Next() {
        mask.x = static_cast<int32_t>((tSide.x < tSide.y) & (tSide.x <= tSide.z));
        mask.y = static_cast<int32_t>((tSide.y < tSide.z) & (tSide.y <= tSide.x));
        mask.z = static_cast<int32_t>((tSide.z < tSide.x) & (tSide.z <= tSide.y));
        tNext = mask.x ? tSide.x : mask.y ? tSide.y : mask.z ? tSide.z : INFINITY;
    }

    KOUEK_CUDA_HOST_DEV void Step() {
        tCurr = tNext;
        tSide.x = isinf(tDlt.x) ? INFINITY : mask.x ? tSide.x + tDlt.x : tSide.x;
        tSide.y = isinf(tDlt.y) ? INFINITY : mask.y ? tSide.y + tDlt.y : tSide.y;
        tSide.z = isinf(tDlt.z) ? INFINITY : mask.z ? tSide.z + tDlt.z : tSide.z;
        chPos += mask * sign;
    }
};

struct KOUEK_CUDA_ALIGN DepthDDA2D {
    glm::vec<3, int32_t> sign;
    glm::vec<3, int32_t> mask;
    glm::vec<3, int32_t> posInBrick;
    float t, tStart;
    float dep;
    float tDlt2Dep;
    glm::vec3 tSide;
    glm::vec3 tDlt;

    KOUEK_CUDA_HOST_DEV bool Init(float t, float maxPosValInBrick, int32_t minDepPosValInBrick,
                                  int32_t maxDepPosValInBrick, const glm::vec3 &posInBrick,
                                  const kouek::Ray &ray) {
        dep = 0.f;
        sign = {ray.dir.x > 0.f   ? 1
                : ray.dir.x < 0.f ? -1
                                  : 0,
                ray.dir.y > 0.f   ? 1
                : ray.dir.y < 0.f ? -1
                                  : 0,
                ray.dir.z > 0.f   ? 1
                : ray.dir.z < 0.f ? -1
                                  : 0};
        this->t = tStart = t;

        glm::ivec3 depSign;
        {
            glm::vec3 distToAxis{sign.x == 0  ? INFINITY
                                 : sign.x > 0 ? posInBrick.x
                                              : maxPosValInBrick - posInBrick.x,
                                 sign.y == 0  ? INFINITY
                                 : sign.y > 0 ? posInBrick.y
                                              : maxPosValInBrick - posInBrick.y,
                                 sign.z == 0  ? INFINITY
                                 : sign.z > 0 ? posInBrick.z
                                              : maxPosValInBrick - posInBrick.z};
            depSign.x = distToAxis.x < distToAxis.y && distToAxis.x <= distToAxis.z ? sign.x : 0;
            depSign.y = distToAxis.y < distToAxis.z && distToAxis.y <= distToAxis.x ? sign.y : 0;
            depSign.z = distToAxis.z < distToAxis.x && distToAxis.z <= distToAxis.y ? sign.z : 0;

            if (depSign.x != 0 && distToAxis.x >= .5f)
                return false;
            if (depSign.y != 0 && distToAxis.y >= .5f)
                return false;
            if (depSign.z != 0 && distToAxis.z >= .5f)
                return false;
        }

        tDlt = glm::abs(1.f / ray.dir);
        this->posInBrick = glm::floor(posInBrick);
        tSide = ((glm::floor(posInBrick) - posInBrick + .5f) * glm::vec3{sign} + .5f) * tDlt + t;

        if (depSign.x != 0) {
            this->posInBrick.x = depSign.x == 1 ? minDepPosValInBrick : maxDepPosValInBrick;
            sign.x = 0;
            tSide.x = INFINITY;
            tDlt2Dep = glm::abs(ray.dir.x);
        }
        if (depSign.y != 0) {
            this->posInBrick.y = depSign.y == 1 ? minDepPosValInBrick : maxDepPosValInBrick;
            sign.y = 0;
            tSide.y = INFINITY;
            tDlt2Dep = glm::abs(ray.dir.y);
        }
        if (depSign.z != 0) {
            this->posInBrick.z = depSign.z == 1 ? minDepPosValInBrick : maxDepPosValInBrick;
            sign.z = 0;
            tSide.z = INFINITY;
            tDlt2Dep = glm::abs(ray.dir.z);
        }

        return (depSign.x | depSign.y | depSign.z);
    }

    KOUEK_CUDA_HOST_DEV void StepNext() {
        mask.x = static_cast<int32_t>((tSide.x < tSide.y) & (tSide.x <= tSide.z));
        mask.y = static_cast<int32_t>((tSide.y < tSide.z) & (tSide.y <= tSide.x));
        mask.z = static_cast<int32_t>((tSide.z < tSide.x) & (tSide.z <= tSide.y));

        t = mask.x ? tSide.x : mask.y ? tSide.y : mask.z ? tSide.z : INFINITY;
        dep = tDlt2Dep * (t - tStart);

        tSide.x = isinf(tDlt.x) ? INFINITY : mask.x ? tSide.x + tDlt.x : tSide.x;
        tSide.y = isinf(tDlt.y) ? INFINITY : mask.y ? tSide.y + tDlt.y : tSide.y;
        tSide.z = isinf(tDlt.z) ? INFINITY : mask.z ? tSide.z + tDlt.z : tSide.z;

        posInBrick += mask * sign;
    }
};

} // namespace RayCaster
} // namespace kouek

#endif // !KOUEK_RAY_CASTER_DDA_H
