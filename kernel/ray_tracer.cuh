#ifndef KOUEK_RAY_TRACER_CUH
#define KOUEK_RAY_TRACER_CUH

#include <glm/glm.hpp>

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <pre_def.hpp>

namespace kouek {

void uploadRenderInfo(const RenderInfo &val);
void uploadSceneInfo(const SceneInfo& val);
void uploadCamera(const glm::mat4 &val);
void uploadProjection(const glm::mat4 &val);

void render(cudaSurfaceObject_t output, const glm::uvec2 &rndrRes,
            bool rndrLowRes, bool reAccumulate);

} // namespace kouek

#endif // !KOUEK_RAY_TRACER_CUH
