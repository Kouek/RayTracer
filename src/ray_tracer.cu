#include "ray_tracer.h"

#include <cuda_gl_interop.h>

#include <cuda/cuda_helper.hpp>

#include <ray_tracer.cuh>

void kouek::RayTracer::Init() {
    int devCnt = 0;
    cudaGetDeviceCount(&devCnt);

    auto outputTexXDDim = [](int8_t x, const cudaDeviceProp &prop) {
        std::cout << ">> max texture" << static_cast<char>(x + '0')
                  << "D dim: ";
        if (x == 1) {
            std::cout << prop.maxTexture1D << std::endl;
            return;
        }
        auto arr = x == 2 ? prop.maxTexture2D : prop.maxTexture3D;
        for (int8_t d = 0; d < x; ++d) {
            if (d != 0)
                std::cout << ", ";
            std::cout << arr[d];
        }
        std::cout << std::endl;
    };
    for (int dev = 0; dev < devCnt; ++dev) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);
        std::cout << "CUDA device: " << prop.name << std::endl;
        for (uint8_t x = 1; x <= 3; ++x)
            outputTexXDDim(x, prop);
    }
}

void kouek::RayTracer::Release() {
    if (outputResc) {
        CHECK_CUDA(cudaGraphicsUnregisterResource(outputResc));
        outputResc = nullptr;
    }
}

void kouek::RayTracer::SetOutput(GLuint tex, const glm::uvec2 &res,
                                 uint8_t lowResLOD) {
    memset(&outputDesc, 0, sizeof(outputDesc));
    outputDesc.resType = cudaResourceTypeArray;
    CHECK_CUDA(
        cudaGraphicsGLRegisterImage(&outputResc, tex, GL_TEXTURE_2D,
                                    cudaGraphicsRegisterFlagsWriteDiscard));

    rndrInfo[&RenderInfo::res] = res;
    lowResScale = 1.f / (float)(1 << lowResLOD);
    rndrInfo[&RenderInfo::lowRes] = glm::vec2{res} * lowResScale;
}

void kouek::RayTracer::SetMaxDepth(glm::uint maxDepth) {
    rndrInfo[&RenderInfo::maxDepth] = maxDepth;
}

void kouek::RayTracer::Prepare() {
    rndrInfo.Update([&]() { uploadRenderInfo(rndrInfo.val); });
    scn.Update([&]() { uploadSceneInfo(scn()->GetScnInfo()); });
    proj.Update([&]() { uploadProjection(proj.val); });
    
    auto lastRndrLowRes = rndrLowRes;
    rndrLowRes = false;
    cam.Update([&]() {
        uploadCamera(cam.val);
        rndrLowRes = true;
    });

    reAccumulate = false;
    if (rndrLowRes != lastRndrLowRes)
        reAccumulate = true;

}

float kouek::RayTracer::Render() {
    cudaGraphicsMapResources(1, &outputResc);
    cudaGraphicsSubResourceGetMappedArray(&outputDesc.res.array.array,
                                          outputResc, 0, 0);
    cudaSurfaceObject_t outputSurf;
    cudaCreateSurfaceObject(&outputSurf, &outputDesc);

    render(outputSurf,
           rndrLowRes ? rndrInfo(&RenderInfo::lowRes)
                      : rndrInfo(&RenderInfo::res),
           rndrLowRes, reAccumulate);

    cudaDestroySurfaceObject(outputSurf);
    cudaGraphicsUnmapResources(1, &outputResc);

    return rndrLowRes ? lowResScale : 1.f;
}
