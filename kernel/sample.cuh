#ifndef KOUEK_SAMPLE_CUH
#define KOUEK_SAMPLE_CUH

#include <cuda.h>
#include <device_launch_parameters.h>

#include <glm/glm.hpp>

#include <cg/mesh.hpp>

#include "common.cuh"

namespace kouek {

__device__ glm::vec3 emitterSampling(const State &state,
                                     const glm::vec3 &rayPos,
                                     const glm::vec3 &rayDir,
                                     const LightSample &lightSample,
                                     const BSDFSample &bsdfSample);
__device__ LightSample lightSampling(const Mesh::Light &light,
                                     glm::uint lightNum);
__device__ glm::vec3 PhongSampling(State &inoutState, const Mesh::Material &mtl,
                                   const glm::vec3 &rayDir);
__device__ float PhongPDF(const State &state, const Mesh::Material &mtl,
                          const glm::vec3 &rayDir, const glm::vec3 &bsdfDir);
__device__ glm::vec3 PhongEval(const State &state, const Mesh::Material &mtl,
                               const glm::vec3 &rayDir,
                               const glm::vec3 &bsdfDir);

} // namespace kouek

#endif // !KOUEK_SAMPLE_CUH
