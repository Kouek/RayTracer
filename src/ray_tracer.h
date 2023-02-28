#ifndef KOUEK_RAY_TRACER_H
#define KOUEK_RAY_TRACER_H

#include <memory>

#include <cuda.h>
#include <cuda_runtime.h>

#include <glad/glad.h>

#include <glm/glm.hpp>

#include <pre_def.hpp>

#include "ray_trace_scn.h"

namespace kouek {

class RayTracer {
  private:
    Updatable<RenderInfo> rndrInfo;
    Updatable<glm::mat4> proj;
    Updatable<glm::mat4> cam;
    Updatable<std::shared_ptr<RayTraceScn>> scn;

    cudaResourceDesc outputDesc;
    cudaGraphicsResource_t outputResc = nullptr;

  public:
    void Init();
    void Release();
    inline void SetScene(std::shared_ptr<RayTraceScn> ptr) { scn = ptr; }
    inline void SetProjection(const glm::mat4 &val) { proj = val; }
    inline void SetCamera(const glm::mat4 &val) { cam = val; }
    void SetOutput(GLuint tex, const glm::uvec2 &res);
    void Prepare();
    void Render();

  private:
};

} // namespace kouek

#endif // !KOUEK_RAY_TRACER_H
