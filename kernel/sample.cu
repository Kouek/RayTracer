#include "sample.cuh"

#include <curand_kernel.h>

#include <pre_def.hpp>

static constexpr auto Ns2SpecularRatio = .001f; // 1/1000

extern __constant__ curandState *dc_curandStatePtr;

inline __device__ glm::vec3 hemisphereCosSampling(float u0, float u1) {
    auto r = glm::sqrt(u0);
    auto phi = 2.f * glm::pi<float>() * u1;
    glm::vec3 dir;
    dir.x = r * glm::cos(phi);
    dir.y = r * glm::sin(phi);
    dir.z = glm::sqrt(glm::max(0.f, 1.f - dir.x * dir.x - dir.y * dir.y));

    return dir;
}

__device__ glm::vec3 kouek::emitterSampling(const State &state,
                                            const glm::vec3 &rayPos,
                                            const glm::vec3 &rayDir,
                                            const LightSample &lightSample,
                                            const BSDFSample &bsdfSample) {
    auto Le = glm::zero<glm::vec3>();
    if (state.depth == 0 || state.isSpecular)
        Le = lightSample.emission;
    else
        Le = powerHeuristic(bsdfSample.pdf, lightSample.pdf) *
             lightSample.emission;
    return Le;
}

__device__ kouek::LightSample kouek::lightSampling(const Mesh::Light &light,
                                                   glm::uint lightNum) {
    auto ret = LightSample::Create();
    ret.emission = light.radiance * (float)lightNum;

    auto r0 = curand_uniform(dc_curandStatePtr);
    auto r1 = curand_uniform(dc_curandStatePtr);

    if (light.type == Mesh::Light::Type::Quad) {
        ret.surfacePos =
            light.quad.pos + r0 * light.quad.uv[0] + r1 * light.quad.uv[1];
        ret.normal =
            glm::normalize(glm::cross(light.quad.uv[0], light.quad.uv[1]));
    } else {
        ret.surfacePos = light.sphere.pos +
                         light.sphere.radius * hemisphereCosSampling(r0, r1);
        ret.normal = glm::normalize(ret.surfacePos - light.sphere.pos);
    }

    return ret;
}

__device__ glm::vec3 kouek::PhongSampling(State &inoutState,
                                          const Mesh::Material &mtl,
                                          const glm::vec3 &rayDir) {
    auto N = inoutState.isFront ? inoutState.normal : -inoutState.normal;
    auto V = -rayDir;

    auto r0 = curand_uniform(dc_curandStatePtr);
    auto r1 = curand_uniform(dc_curandStatePtr);
    auto probability = curand_uniform(dc_curandStatePtr);

    auto specularRatio = mtl.ns * Ns2SpecularRatio;
    auto diffuseRatio = 1.f - specularRatio;

    auto TB = computeTangent(N);
    glm::vec3 dir;
    if (probability < diffuseRatio) {
        inoutState.isSpecular = false;

        dir = hemisphereCosSampling(r0, r1);
        dir = dir.x * TB.T + dir.y * TB.B + dir.z * N;
    } else {
        inoutState.isSpecular = true;

        auto specularAlpha = glm::max(Ns2SpecularRatio, specularRatio);
        auto phi = 2.f * glm::pi<float>() * r0;
        auto cosTheta = glm::sqrt(
            (1.f - r1) / (1.f + (specularAlpha * specularAlpha - 1.f) * r1));
        auto sinTheta =
            glm::clamp(glm::sqrt(1.f - (cosTheta * cosTheta)), 0.f, 1.f);
        auto sinPhi = glm::sin(phi);
        auto cosPhi = glm::cos(phi);

        auto halfVec =
            glm::vec3{sinTheta * cosPhi, sinTheta * sinPhi, cosTheta};
        halfVec = halfVec.x * TB.T + halfVec.y * TB.B + halfVec.z * N;

        dir = 2.f * glm::dot(V, halfVec) * halfVec - V;
    }
    return dir;
}

__device__ float kouek::PhongPDF(const State &state, const Mesh::Material &mtl,
                                 const glm::vec3 &rayDir,
                                 const glm::vec3 &bsdfDir) {
    auto N = state.isFront ? state.normal : -state.normal;
    auto V = -rayDir;

    auto specularRatio = mtl.ns * Ns2SpecularRatio;
    auto diffuseRatio = 1.f - specularRatio;
    auto specularAlpha = glm::max(Ns2SpecularRatio, specularRatio);

    auto halfVec = glm::normalize(bsdfDir + V);

    float cosTheta = glm::abs(glm::dot(halfVec, N));
    float pdfGTR2 = GTR2(cosTheta, specularAlpha) * cosTheta;

    float pdfSpec = pdfGTR2 / (4.0 * abs(dot(bsdfDir, halfVec)));
    float pdfDiff = glm::abs(glm::dot(bsdfDir, N)) * (1.f / glm::pi<float>());

    return diffuseRatio * pdfDiff + specularRatio * pdfSpec;
}

__device__ glm::vec3 kouek::PhongEval(const State &state,
                                      const Mesh::Material &mtl,
                                      const glm::vec3 &rayDir,
                                      const glm::vec3 &bsdfDir) {
    return glm::vec3{1.f};
}
