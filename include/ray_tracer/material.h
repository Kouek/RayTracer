#ifndef KOUEK_RAY_TRACER_MATERIAL_H
#define KOUEK_RAY_TRACER_MATERIAL_H

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

#include <cg/math.h>

namespace kouek {
namespace RayTracer {

struct KOUEK_CUDA_ALIGN Material {
    bool emissive;
    glm::vec3 emission;
    glm::vec3 kd;
    glm::vec3 ks;
    glm::vec3 tr;
    float ni;
    float ns;

    float kdOverKdAddKs;

    KOUEK_CUDA_HOST_DEV static Material Create(const glm::vec3 &kd, const glm::vec3 &ks,
                                               const glm::vec3 &tr, float ni, float ns,
                                               const glm::vec3 &emission = glm::vec3(0.f)) {
        Material matr;
        matr.emissive = emission != glm::vec3(0.f);
        matr.emission = emission;
        matr.kd = kd;
        matr.ks = ks;
        matr.tr = tr;
        matr.ns = ns;
        matr.ni = ni;
        auto kdAvg = (kd.r + kd.g + kd.b) / 3.f;
        auto ksAvg = (ks.r + ks.g + ks.b) / 3.f;
        matr.kdOverKdAddKs = kdAvg / (kdAvg + ksAvg);

        return matr;
    }

    KOUEK_CUDA_HOST_DEV glm::vec3 BRDF(const glm::vec3 &dirIn, const glm::vec3 &dirOut,
                                       const glm::vec3 &norm) const {
        auto diffuse = kd * glm::one_over_pi<float>();
        auto reflect = glm::normalize(2.f * glm::dot(norm, dirOut) * norm - dirOut);
        auto specular = ks * (ns + 2.f) * glm::one_over_two_pi<float>() *
                        glm::pow(glm::max(glm::dot(dirIn, reflect), 0.f), ns);
        return diffuse + specular;
    }
};

struct KOUEK_CUDA_ALIGN Light {
    enum class Type : uint8_t { Quad, Sphere };

    Type type;
    union {
        struct {
            glm::vec3 o, u, v, n;
            float invArea;
        } quad;
        struct {
            glm::vec3 o;
            float r, rSqr;
        } sphere;
    };
    float area;
    glm::vec3 radiance;

    KOUEK_CUDA_HOST_DEV static Light CreateQuad(const glm::vec3 &o, const glm::vec3 &u,
                                                const glm::vec3 &v, const glm::vec3 &radiance) {
        Light lht;
        lht.type = Type::Quad;
        lht.quad.o = o;
        lht.quad.u = u;
        lht.quad.v = v;
        lht.quad.n = glm::normalize(glm::cross(lht.quad.u, lht.quad.v));
        lht.area = glm::length(u) * glm::length(v);
        lht.quad.invArea = 1.f / lht.area;
        lht.radiance = radiance;

        return lht;
    }
    KOUEK_CUDA_HOST_DEV static Light CreateSphere(float r, const glm::vec3 &o,
                                                  const glm::vec3 &radiance) {
        Light lht;
        lht.type = Type::Sphere;
        lht.sphere.o = o;
        lht.sphere.r = r;
        lht.sphere.rSqr = r * r;
        lht.area = 2.f * glm::two_pi<float>() * lht.sphere.rSqr;
        lht.radiance = radiance;

        return lht;
    }

    struct SampleResult {
        glm::vec3 pos;
        glm::vec3 norm;
        float pdf;
    };
    KOUEK_CUDA_HOST_DEV SampleResult SampleQuad(float u0, float u1) const {
        return {quad.o + u0 * quad.u + u1 * quad.v, quad.n, quad.invArea};
    }
    KOUEK_CUDA_HOST_DEV SampleResult SampleSphere(float u0, float u1, const glm::vec3 &pos) const {
        constexpr auto UseUniformHemisphereSampling = true;
        if constexpr (UseUniformHemisphereSampling) {
            auto dirO2P = glm::normalize(pos - sphere.o);
            auto tngnt = Math::GenerateTangent(dirO2P);

            SampleResult ret;
            ret.norm = kouek::Math::ThetaPhiToDirection(glm::acos(1 - u0),
                                                        glm::two_pi<float>() * u1, dirO2P, tngnt);
            ret.pos = sphere.o + sphere.r * ret.norm;
            ret.pdf = glm::one_over_two_pi<float>();
            return ret;
        } else {
            auto p2o = sphere.o - pos;
            auto distSqrP2O = glm::dot(p2o, p2o);
            auto cosThetaMax = glm::sqrt(glm::max(1.f - sphere.rSqr / distSqrP2O, 0.f));
            auto cosTheta = (1.f - u0) + u0 * cosThetaMax;
            auto phi = glm::two_pi<float>() * u1;

            auto distP2O = glm::sqrt(distSqrP2O);
            auto sinThetaSqr = 1.f - cosTheta * cosTheta;
            auto distP2X = distP2O * cosTheta -
                           glm::sqrt(glm::max(sphere.rSqr - distSqrP2O * sinThetaSqr, 0.f));
            auto cosAlpha =
                (distSqrP2O + sphere.rSqr - distP2X * distP2X) / (2.f * distP2O * sphere.r);

            SampleResult ret;
            ret.norm = glm::normalize(-p2o);
            auto tngnt = Math::GenerateTangent(ret.norm);
            ret.norm = Math::CosineThetaPhiToDirection(cosAlpha, phi, ret.norm, tngnt);

            ret.pos = sphere.o + sphere.r * ret.norm;
            ret.pdf = glm::one_over_two_pi<float>() / (1.f - cosThetaMax);

            return ret;
        }
    }
};

} // namespace RayTracer
} // namespace kouek

#endif // !KOUEK_RAY_TRACER_MATERIAL_H
