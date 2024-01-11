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
#define CHECK_CUDA(call) call
#else
#define CHECK_CUDA(call) Check(call, __FILE__, __LINE__)
#endif // NDEBUG

#ifdef __CUDACC__
#define __host_dev__ __host__ __device__
#else
#define __host_dev__
#endif // __CUDACC__

constexpr auto FloatMin = std::numeric_limits<float>::lowest();
constexpr auto FloatMax = std::numeric_limits<float>::max();

} // namespace CUDA
} // namespace kouek

#endif // !KOUEK_CUDA_HELPER_H
