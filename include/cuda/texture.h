#ifndef KOUEK_CUDA_TEXTURE_H
#define KOUEK_CUDA_TEXTURE_H

#include <optional>

#include <vector>

#include <cuda/helper.h>

#include <glm/glm.hpp>

#include <util/meta_type.h>

namespace kouek {
namespace CUDA {

class Array : Noncopyable {
  private:
    bool isComplete = false;
    cudaArray_t arr = nullptr;

  public:
    Array(const cudaChannelFormatDesc &chnDesc, const glm::vec<3, size_t> &dim) {
        if (dim.z != 0) {
            auto extent = make_cudaExtent(dim.x, dim.y, dim.z);
            isComplete = cudaSuccess == KOUEK_CUDA_CHECK(cudaMalloc3DArray(&arr, &chnDesc, extent));
        } else
            isComplete =
                cudaSuccess == KOUEK_CUDA_CHECK(cudaMallocArray(&arr, &chnDesc, dim.x, dim.y));
    }
    template <typename T>
    Array(const std::vector<T> &dat, const glm::vec<3, size_t> &dim,
          const std::optional<cudaChannelFormatDesc> &chnDescOpt = {}) {
        auto chnDesc = chnDescOpt.value_or(cudaCreateChannelDesc<T>());
        if (dim.z != 0) {
            auto extent = make_cudaExtent(dim.x, dim.y, dim.z);
            isComplete = cudaSuccess == KOUEK_CUDA_CHECK(cudaMalloc3DArray(&arr, &chnDesc, extent));
            if (!isComplete)
                return;

            cudaMemcpy3DParms param;
            memset(&param, 0, sizeof(param));
            param.srcPtr =
                make_cudaPitchedPtr((void *)(dat.data()), sizeof(T) * dim.x, dim.x, dim.y);
            param.extent = extent;
            param.dstArray = arr;
            param.kind = cudaMemcpyHostToDevice;
            isComplete &= cudaSuccess == KOUEK_CUDA_CHECK(cudaMemcpy3D(&param));
        } else {
            isComplete =
                cudaSuccess == KOUEK_CUDA_CHECK(cudaMallocArray(&arr, &chnDesc, dim.x, dim.y));
            if (!isComplete)
                return;

            isComplete &= cudaSuccess ==
                          KOUEK_CUDA_CHECK(cudaMemcpyToArray(
                              arr, 0, 0, dat.data(), sizeof(T) * dim.x * std::max(size_t(1), dim.y),
                              cudaMemcpyHostToDevice));
        }
    }
    ~Array() {
        if (isComplete) {
            KOUEK_CUDA_CHECK(cudaFreeArray(arr));
            arr = nullptr;
            isComplete = false;
        }
    }

    bool IsComplete() const { return isComplete; }
    cudaArray_t Get() const { return arr; }
    cudaExtent GetExtent() const {
        cudaChannelFormatDesc chnDesc;
        cudaExtent extent;
        KOUEK_CUDA_CHECK(cudaArrayGetInfo(&chnDesc, &extent, nullptr, arr));

        return extent;
    }
};

class Texture : Noncopyable {
  private:
    bool isComplete = false;
    cudaTextureObject_t tex = 0;

    std::shared_ptr<Array> arr;

  public:
    Texture(std::shared_ptr<Array> arr, const std::optional<cudaTextureDesc> &texDescOpt = {}) {
        cudaTextureDesc texDesc;
        if (texDescOpt.has_value())
            texDesc = texDescOpt.value();
        else {
            memset(&texDesc, 0, sizeof(texDesc));
            texDesc.normalizedCoords = 0;
            texDesc.filterMode = cudaFilterModeLinear;
            texDesc.addressMode[0] = texDesc.addressMode[1] = texDesc.addressMode[2] =
                cudaAddressModeBorder;
            texDesc.readMode = cudaReadModeNormalizedFloat;
        }

        this->arr = arr;

        cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = this->arr->Get();

        isComplete = cudaSuccess ==
                     KOUEK_CUDA_CHECK(cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr));
    }
    ~Texture() {
        if (isComplete) {
            KOUEK_CUDA_CHECK(cudaDestroyTextureObject(tex));
            isComplete = false;
        }
    }

    bool IsComplete() const { return isComplete; }
    cudaTextureObject_t Get() const { return tex; }
    const Array *GetArray() const { return arr.get(); }
};

class Surface : Noncopyable {
  private:
    bool isComplete = false;
    cudaSurfaceObject_t surf = 0;

    std::shared_ptr<Array> arr;

  public:
    Surface(std::shared_ptr<Array> arr) {
        this->arr = arr;

        cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = this->arr->Get();

        isComplete = cudaSuccess == KOUEK_CUDA_CHECK(cudaCreateSurfaceObject(&surf, &resDesc));
    }
    ~Surface() {
        if (isComplete) {
            KOUEK_CUDA_CHECK(cudaDestroySurfaceObject(surf));
            isComplete = false;
        }
    }

    bool IsComplete() const { return isComplete; }
    cudaSurfaceObject_t Get() const { return surf; }
};

} // namespace CUDA
} // namespace kouek

#endif // !KOUEK_CUDA_TEXTURE_H
