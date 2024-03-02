#ifndef KOUEK_CUDA_HELPER_H
#define KOUEK_CUDA_HELPER_H

#include <format>
#include <iostream>
#include <limits>

#include <cuda.h>
#include <cuda_runtime.h>

namespace kouek {
namespace CUDA {

inline cudaError_t Check(cudaError_t err, const char *fileName, uint32_t line) {
    if (err == cudaSuccess)
        return err;
    std::cerr << std::format("[CUDA Error] at {}:{}. {}: {}\n", fileName, line,
                             cudaGetErrorName(err), cudaGetErrorString(err));
    return err;
}

#ifdef NDEBUG
#define KOUEK_CUDA_CHECK(call) call
#else
#define KOUEK_CUDA_CHECK(call) kouek::CUDA::Check(call, __FILE__, __LINE__)
#endif // NDEBUG

#ifdef __CUDACC__
#define KOUEK_CUDA_HOST_DEV __host__ __device__
#define KOUEK_CUDA_ALIGN __align__(16)
#else
#define KOUEK_CUDA_HOST_DEV
#define KOUEK_CUDA_ALIGN alignas(16)
#endif // __CUDACC__

constexpr auto FloatMin = std::numeric_limits<float>::lowest();
constexpr auto FloatMax = std::numeric_limits<float>::max();
constexpr auto FloatEps = std::numeric_limits<float>::epsilon();

} // namespace CUDA
} // namespace kouek

#endif // !KOUEK_CUDA_HELPER_H
