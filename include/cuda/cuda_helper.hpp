#ifndef KOUEK_CUDA_HELPER_H
#define KOUEK_CUDA_HELPER_H

#include <iostream>

#include <cuda.h>

namespace kouek {

inline void checkCUDA(cudaError_t result, const char *const file,
                      int const line) {
    if (result != cudaSuccess) {
        auto errName = cudaGetErrorName(result);
        auto errStr = cudaGetErrorString(result);
        std::cout << "[ERROR] CUDA err: " << errName << '('
                  << static_cast<unsigned int>(result) << ": " << errStr
                  << "), at " << file << ':' << line << std::endl;
    }
}

} // namespace kouek

#define CHECK_CUDA(call) kouek::checkCUDA(call, __FILE__, __LINE__);

#endif // !KOUEK_CUDA_HELPER_H
