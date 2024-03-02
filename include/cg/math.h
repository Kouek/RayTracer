#ifndef KOUEK_MATH_H
#define KOUEK_MATH_H

#include <format>
#include <iostream>

#include <array>

#include <glm/gtc/matrix_transform.hpp>

namespace kouek {
namespace Math {

inline glm::mat4 InverseProjective(const glm::mat4 &projection) {
    // partition
    // [ A | B ]
    // ----+----
    // [ C | D ]
    glm::mat2 a(projection[0][0], projection[0][1], projection[1][0], projection[1][1]);
    glm::mat2 b(projection[2][0], projection[2][1], projection[3][0], projection[3][1]);
    glm::mat2 c(projection[0][2], projection[0][3], projection[1][2], projection[1][3]);
    glm::mat2 d(projection[2][2], projection[2][3], projection[3][2], projection[3][3]);

    // pre-compute repeated parts
    a = glm::inverse(a);      // invA
    glm::mat2 ab = a * b;     // invA * B
    glm::mat2 ca = c * a;     // C * invA
    glm::mat2 cab = ca * b;   // C * invA * B
    glm::mat2 dcab = d - cab; // D * C * invA * B

    float det = dcab[1][1] * dcab[0][0] - dcab[1][0] * dcab[0][1];
    if (fabsf(det) <= std::numeric_limits<float>::epsilon())
        return glm::identity<glm::mat4>();

    // compute D' and -D'
    glm::mat2 d1 = glm::inverse(dcab);
    glm::mat2 d2 = -d1;
    // compute C'
    glm::mat2 c1 = d2 * ca; // -D' * C * invA
    // compute B'
    glm::mat2 b1 = ab * d2; // invA * B * (-D')
    // compute A'
    glm::mat2 a1 = a - (ab * c1); // invA - invA * B * C'

    // composition
    // [ A'| B']
    // ----+----
    // [ C'| D']
    return glm::mat4(a1[0][0], a1[0][1], c1[0][0], c1[0][1], a1[1][0], a1[1][1], c1[1][0], c1[1][1],
                     b1[0][0], b1[0][1], d1[0][0], d1[0][1], b1[1][0], b1[1][1], d1[1][0],
                     d1[1][1]);
}

inline glm::mat4 InversePose(const glm::mat4 &pose) {
    return glm::mat4(pose[0][0], pose[1][0], pose[2][0], 0, pose[0][1], pose[1][1], pose[2][1], 0,
                     pose[0][2], pose[1][2], pose[2][2], 0,
                     -(pose[0][0] * pose[3][0] + pose[0][1] * pose[3][1] + pose[0][2] * pose[3][2]),
                     -(pose[1][0] * pose[3][0] + pose[1][1] * pose[3][1] + pose[1][2] * pose[3][2]),
                     -(pose[2][0] * pose[3][0] + pose[2][1] * pose[3][1] + pose[2][2] * pose[3][2]),
                     1.f);
}

inline void PrintGLMMat4(const glm::mat4 &mat4, const char *name = nullptr) {
    if (name == nullptr)
        std::cout << std::format("[\n{}\t{}\t{}\t{}\n", mat4[0][0], mat4[1][0], mat4[2][0],
                                 mat4[3][0]);
    else
        std::cout << std::format("{}:[\n{}\t{}\t{}\t{}\n", name, mat4[0][0], mat4[1][0], mat4[2][0],
                                 mat4[3][0]);
    std::cout << std::format("{}\t{}\t{}\t{}\n", mat4[0][1], mat4[1][1], mat4[2][1], mat4[3][1]);
    std::cout << std::format("{}\t{}\t{}\t{}\n", mat4[0][2], mat4[1][2], mat4[2][2], mat4[3][2]);
    std::cout << std::format("{}\t{}\t{}\t{}\n]\n", mat4[0][3], mat4[1][3], mat4[2][3], mat4[3][3]);
}

#ifdef __CUDA_ARCH__
__device__
#endif
    inline void
    HDRToLDRCorrect(glm::vec3 &rgb) {
    rgb = rgb / (rgb + 1.f);
}

#ifdef __CUDA_ARCH__
__device__
#endif
    inline void
    GammaCorrect(glm::vec3 &rgb) {
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
    for (uint8_t xyz = 0; xyz < 3; ++xyz)
        rgb[xyz] = glm::pow(rgb[xyz], 1.f / 2.2f);
}

#ifdef __CUDA_ARCH__
__device__
#endif
    inline glm::vec3
    GenerateTangent(const glm::vec3 &norm) {
    if (glm::abs(norm.y) < .999f)
        return glm::normalize(glm::cross(norm, glm::vec3(0.f, 1.f, 0.f)));
    else
        return glm::normalize(glm::cross(norm, glm::vec3(1.f, 0.f, 0.f)));
}

#ifdef __CUDA_ARCH__
__device__
#endif
    inline glm::vec3
    ThetaPhiToDirection(float theta, float phi, const glm::vec3 &norm, const glm::vec3 &tngnt) {
    auto sinTheta = glm::sin(theta);
    glm::vec3 dir = {glm::cos(phi) * sinTheta, glm::cos(theta), glm::sin(phi) * sinTheta};

    auto biTngnt = glm::normalize(glm::cross(tngnt, norm));
    return glm::normalize(glm::mat3(tngnt, norm, biTngnt) * dir);
}

#ifdef __CUDA_ARCH__
__device__
#endif
    inline glm::vec3
    CosineThetaPhiToDirection(float cosTheta, float phi, const glm::vec3 &norm,
                              const glm::vec3 &tngnt) {
    auto sinTheta = glm::sqrt(1.f - cosTheta * cosTheta);
    glm::vec3 dir = {glm::cos(phi) * sinTheta, cosTheta, glm::sin(phi) * sinTheta};

    auto biTngnt = glm::normalize(glm::cross(tngnt, norm));
    return glm::normalize(glm::mat3(tngnt, norm, biTngnt) * dir);
}

} // namespace Math
} // namespace kouek

#endif // !KOUEK_MATH_H
