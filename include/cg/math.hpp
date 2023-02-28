#ifndef KOUEK_MATH_H
#define KOUEK_MATH_H

#include <array>

#include <glm/gtc/matrix_transform.hpp>

namespace kouek {
class Math {
  public:
    static inline glm::mat4 InverseProjective(const glm::mat4 &projection) {
        // partition
        // [ A | B ]
        // ----+----
        // [ C | D ]
        glm::mat2 a(projection[0][0], projection[0][1], projection[1][0],
                    projection[1][1]);
        glm::mat2 b(projection[2][0], projection[2][1], projection[3][0],
                    projection[3][1]);
        glm::mat2 c(projection[0][2], projection[0][3], projection[1][2],
                    projection[1][3]);
        glm::mat2 d(projection[2][2], projection[2][3], projection[3][2],
                    projection[3][3]);

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
        return glm::mat4(a1[0][0], a1[0][1], c1[0][0], c1[0][1], a1[1][0],
                         a1[1][1], c1[1][0], c1[1][1], b1[0][0], b1[0][1],
                         d1[0][0], d1[0][1], b1[1][0], b1[1][1], d1[1][0],
                         d1[1][1]);
    }

    static inline glm::mat4 InversePose(const glm::mat4 &pose) {
        return glm::mat4(pose[0][0], pose[1][0], pose[2][0], 0, pose[0][1],
                         pose[1][1], pose[2][1], 0, pose[0][2], pose[1][2],
                         pose[2][2], 0,
                         -(pose[0][0] * pose[3][0] + pose[0][1] * pose[3][1] +
                           pose[0][2] * pose[3][2]),
                         -(pose[1][0] * pose[3][0] + pose[1][1] * pose[3][1] +
                           pose[1][2] * pose[3][2]),
                         -(pose[2][0] * pose[3][0] + pose[2][1] * pose[3][1] +
                           pose[2][2] * pose[3][2]),
                         1.f);
    }

    static inline void PrintGLMMat4(const glm::mat4 &mat4,
                                    const char *name = nullptr) {
        if (name == nullptr)
            printf("[\n%f\t%f\t%f\t%f\n", mat4[0][0], mat4[1][0], mat4[2][0],
                   mat4[3][0]);
        else
            printf("%s:[\n%f\t%f\t%f\t%f\n", name, mat4[0][0], mat4[1][0],
                   mat4[2][0], mat4[3][0]);
        printf("%f\t%f\t%f\t%f\n", mat4[0][1], mat4[1][1], mat4[2][1],
               mat4[3][1]);
        printf("%f\t%f\t%f\t%f\n", mat4[0][2], mat4[1][2], mat4[2][2],
               mat4[3][2]);
        printf("%f\t%f\t%f\t%f\n]\n", mat4[0][3], mat4[1][3], mat4[2][3],
               mat4[3][3]);
    }
};
struct Frustum {
    /// <summary>
    /// Indicies [0,5] represent Back(Near), Front(Far),
    /// Left, Right, Bottom and Top faces
    /// </summary>
    std::array<std::array<float, 4>, 6> coeffs;

    Frustum(const glm::vec3 &startPos, const glm::vec3 &forward, float n,
            float f, const glm::vec3 &LBDrc, const glm::vec3 &LTDrc,
            const glm::vec3 &RBDrc, const glm::vec3 &RTDrc) {
        auto norm = -forward;
        auto pos = startPos + n * forward;
        coeffs[0][0] = norm.x;
        coeffs[0][1] = norm.y;
        coeffs[0][2] = norm.z;
        coeffs[0][3] = -glm::dot(norm, pos);

        norm *= -1.f;
        pos = startPos + f * forward;
        coeffs[1][0] = norm.x;
        coeffs[1][1] = norm.y;
        coeffs[1][2] = norm.z;
        coeffs[1][3] = -glm::dot(norm, pos);

        norm = glm::cross(LTDrc, LBDrc);
        pos = startPos;
        coeffs[2][0] = norm.x;
        coeffs[2][1] = norm.y;
        coeffs[2][2] = norm.z;
        coeffs[2][3] = -glm::dot(norm, pos);

        norm = glm::cross(RBDrc, RTDrc);
        coeffs[3][0] = norm.x;
        coeffs[3][1] = norm.y;
        coeffs[3][2] = norm.z;
        coeffs[3][3] = -glm::dot(norm, pos);

        norm = glm::cross(LBDrc, RBDrc);
        coeffs[4][0] = norm.x;
        coeffs[4][1] = norm.y;
        coeffs[4][2] = norm.z;
        coeffs[4][3] = -glm::dot(norm, pos);

        norm = glm::cross(RTDrc, LTDrc);
        coeffs[5][0] = norm.x;
        coeffs[5][1] = norm.y;
        coeffs[5][2] = norm.z;
        coeffs[5][3] = -glm::dot(norm, pos);
    }
    inline bool IsIntersetcedWith(const glm::vec3 &pos) const {
        bool intersected = true;
        for (uint8_t faceIdx = 0; faceIdx < 6; ++faceIdx) {
            intersected &=
                (coeffs[faceIdx][0] * pos.x + coeffs[faceIdx][1] * pos.y +
                     coeffs[faceIdx][2] * pos.z + coeffs[faceIdx][3] <=
                 0);
            if (!intersected)
                break;
        }
        return intersected;
    }
    inline bool IsIntersectedWithAABB(const glm::vec3 &min,
                                      const glm::vec3 &max) const {
        auto compute = [&](uint8_t faceIdx, const glm::vec3 &pos) {
            return coeffs[faceIdx][0] * pos.x + coeffs[faceIdx][1] * pos.y +
                   coeffs[faceIdx][2] * pos.z + coeffs[faceIdx][3];
        };
        glm::vec3 farPos;
        for (uint8_t faceIdx = 0; faceIdx < 6; ++faceIdx) {
            farPos.x = coeffs[faceIdx][0] > 0 ? min.x : max.x;
            farPos.y = coeffs[faceIdx][1] > 0 ? min.y : max.y;
            farPos.z = coeffs[faceIdx][2] > 0 ? min.z : max.z;
            if (compute(faceIdx, farPos) > 0)
                return false;
        }
        return true;
    }
};

} // namespace kouek

#endif // !KOUEK_MATH_H
