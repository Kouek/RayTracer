#ifndef KOUEK_COMMON_CUH
#define KOUEK_COMMON_CUH

#include <cuda.h>
#include <device_launch_parameters.h>

#include <glm/gtc/matrix_transform.hpp>

#include <pre_def.hpp>

namespace kouek {

struct State {
    bool isFront;
    bool isSpecular;
    glm::uint depth;
    glm::uint gi;
    glm::uint fi;
    glm::uint li; // only valid when hit a light
    float t;
    glm::vec3 bary;
    glm::vec3 fhp; // face hit pos
    glm::vec3 normal;

    static inline __device__ State Create() {
        State ret;
        ret.isSpecular = false;
        ret.depth = 0;
        ret.li = NoneIdx;
        return ret;
    }
};

struct LightSample {
    glm::vec3 surfacePos;
    glm::vec3 normal;
    glm::vec3 emission;
    float pdf;

    static inline __device__ LightSample Create() {
        LightSample ret;
        ret.pdf = ret.emission.x = ret.emission.y = ret.emission.z = 0.f;
        return ret;
    }
};

struct BSDFSample {
    float pdf;
    glm::vec3 dir;

    static inline __device__ BSDFSample Create() {
        BSDFSample ret;
        ret.pdf = 0.f;
        return ret;
    }
};

inline __device__ float4 rgbaUChar4ToFloat4(const uchar4 &uc4) {
    static constexpr auto Div = 1.f / 255.f;
    auto f4 = make_float4(Div * uc4.x, Div * uc4.y, Div * uc4.z, Div * uc4.w);
    f4.x = __saturatef(f4.x); // clamp to [0, 1]
    f4.y = __saturatef(f4.y);
    f4.z = __saturatef(f4.z);
    f4.w = __saturatef(f4.w);
    return f4;
}

inline __device__ uchar4 rgbaFloat4ToUChar4(float4 f4) {
    f4.x = __saturatef(f4.x); // clamp to [0, 1]
    f4.y = __saturatef(f4.y);
    f4.z = __saturatef(f4.z);
    f4.w = __saturatef(f4.w);
    f4.x *= 255.f;
    f4.y *= 255.f;
    f4.z *= 255.f;
    f4.w *= 255.f;
    return make_uchar4(f4.x, f4.y, f4.z, f4.w);
}

struct Tangent {
    glm::vec3 T;
    glm::vec3 B;
};
inline __device__ Tangent computeTangent(const glm::vec3 &N) {
    auto U = glm::abs(N.z) < 0.999f ? glm::vec3{0.f, 0.f, 1.f}
                                    : glm::vec3{1.f, 0.f, 0.f};
    Tangent ret;
    ret.T = glm::normalize(glm::cross(U, N));
    ret.B = glm::cross(N, ret.T);

    return ret;
}

inline __device__ float powerHeuristic(float a, float b) {
    auto t = a * a;
    return t / (b * b + t);
}

inline __device__ float GTR2(float NDotH, float a) {
    float a2 = a * a;
    float t = 1.f + (a2 - 1.f) * NDotH * NDotH;
    return a2 / (glm::pi<float>() * t * t);
}

inline __device__ float SchlickFresnel(float u) {
    float m = glm::clamp(1.f - u, 0.f, 1.f);
    float m2 = m * m;
    return m2 * m2 * m; // pow(m,5)
}

inline __device__ float SmithG_GGX(float NDotv, float alphaG) {
    float a = alphaG * alphaG;
    float b = NDotv * NDotv;
    return 1.f / (NDotv + glm::sqrt(a + b - a * b));
}

inline __device__ bool rayIntersectAABB(float &tMinOut, float &tMaxOut,
                                        const glm::vec3 &rayPos,
                                        const glm::vec3 &rayDir,
                                        const glm::vec3 &minPos,
                                        const glm::vec3 &maxPos) {
    register float ts[8];
    ts[0] = (minPos.x - rayPos.x) / rayDir.x;
    ts[1] = (maxPos.x - rayPos.x) / rayDir.x;
    ts[2] = (minPos.y - rayPos.y) / rayDir.y;
    ts[3] = (maxPos.y - rayPos.y) / rayDir.y;
    ts[4] = (minPos.z - rayPos.z) / rayDir.z;
    ts[5] = (maxPos.z - rayPos.z) / rayDir.z;
    ts[6] = glm::max(glm::max(glm::min(ts[0], ts[1]), glm::min(ts[2], ts[3])),
                     glm::min(ts[4], ts[5]));
    ts[7] = glm::min(glm::min(glm::max(ts[0], ts[1]), glm::max(ts[2], ts[3])),
                     glm::max(ts[4], ts[5]));
    tMinOut = ts[6] < 0.f ? 0.f : ts[6];
    tMaxOut = ts[7];

    return ts[6] <= ts[7];
}

inline __device__ bool rayIntersectTriangle(float &tOut, float &uOut,
                                            float &vOut, float &oneMinusUVOut,
                                            const glm::vec3 &rayPos,
                                            const glm::vec3 &rayDir,
                                            glm::vec3 p3[3]) {
    register auto E1 = p3[1] - p3[0];
    register auto E2 = p3[2] - p3[0];
    register auto S = rayPos - p3[0];

    auto S1 = glm::cross(rayDir, E2);
    auto S2 = glm::cross(S, E1);
    auto inv = 1.f / glm::dot(S1, E1);
    auto t = inv * glm::dot(S2, E2);
    auto u = inv * glm::dot(S1, S);
    auto v = inv * glm::dot(S2, rayDir);
    auto oneMinusUV = 1.f - u - v;
    if (t >= 0.f && u >= 0.f && v >= 0.f && oneMinusUV >= 0.f) {
        tOut = t;
        uOut = u;
        vOut = v;
        oneMinusUVOut = oneMinusUV;
        return true;
    }
    return false;
}

inline __device__ float rayIntersectRect(const glm::vec3 &rayPos,
                                         const glm::vec3 &rayDir,
                                         const glm::vec3 &pos,
                                         const glm::vec3 &N, const glm::vec3 &u,
                                         const glm::vec3 &v) {
    auto DdotN = glm::dot(rayDir, N);
    if (DdotN < Eps)
        return +INFINITY;

    auto t = glm::dot(pos - rayPos, N) / DdotN;
    if (t < 0.f)
        return +INFINITY;

    auto PtoX = rayPos + t * rayDir - pos;
    auto xu = glm::dot(PtoX, u);
    auto xv = glm::dot(PtoX, v);
    auto luSqr = glm::dot(u, u);
    auto lvSqr = glm::dot(v, v);
    if (xu < 0.f || xv < 0.f || xu > luSqr || xv > lvSqr)
        return +INFINITY;

    return t;
}

inline __device__ float rayIntersectSphere(const glm::vec3 &rayPos,
                                           const glm::vec3 &rayDir,
                                           const glm::vec3 &pos, float radius) {
    auto op = pos - rayPos;
    auto b = glm::dot(op, rayDir);
    auto det = b * b - glm::dot(op, op) + radius * radius;
    if (det < 0.f)
        return +INFINITY;

    det = glm::sqrt(det);
    auto t1 = b - det;
    if (t1 > Eps)
        return t1;

    auto t2 = b + det;
    if (t2 > Eps)
        return t2;

    return INFINITY;
}

} // namespace kouek

#endif // !KOUEK_COMMON_CUH
