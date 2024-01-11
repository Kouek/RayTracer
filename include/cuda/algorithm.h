#ifndef KOUEK_CUDA_ALGORITHM_H
#define KOUEK_CUDA_ALGORITHM_H

#include <thrust/device_vector.h>

#include <thrust/scan.h>

namespace kouek {
namespace CUDA {

template <typename IdxTy, typename T>
thrust::device_vector<IdxTy> Difference(const thrust::device_vector<T> &d_srcs, IdxTy srcNum = 0) {
    thrust::device_vector<IdxTy> d_diffs(srcNum == 0 ? d_srcs.size() : srcNum, 1);
    thrust::for_each(thrust::make_counting_iterator(IdxTy(1)),
                     thrust::make_counting_iterator(static_cast<IdxTy>(d_diffs.size())),
                     [diffs = thrust::raw_pointer_cast(d_diffs.data()),
                      srcs = thrust::raw_pointer_cast(d_srcs.data())] __device__(IdxTy srcIdx) {
                         diffs[srcIdx] = srcs[srcIdx - 1] == srcs[srcIdx] ? 0 : 1;
                     });
    return d_diffs;
}

template <typename IdxTy>
thrust::device_vector<IdxTy> CompactIndexes(IdxTy srcNum,
                                            const thrust::device_vector<IdxTy> &d_valids) {
    thrust::device_vector<IdxTy> d_compactedPrefixSums(srcNum);
    thrust::inclusive_scan(d_valids.begin(), d_valids.end(), d_compactedPrefixSums.begin());
    auto cmpctNum = d_compactedPrefixSums.back();

    thrust::device_vector<IdxTy> d_compactedIdxs(cmpctNum);
    thrust::for_each(
        thrust::make_counting_iterator(IdxTy(1)), thrust::make_counting_iterator(srcNum),
        [compactedIdxs = thrust::raw_pointer_cast(d_compactedIdxs.data()),
         compactedPrefixSums =
             thrust::raw_pointer_cast(d_compactedPrefixSums.data())] __device__(IdxTy srcIdx) {
            auto prefixSum = compactedPrefixSums[srcIdx];
            if (prefixSum != compactedPrefixSums[srcIdx - 1])
                compactedIdxs[prefixSum - 1] = srcIdx;
        });
    return d_compactedIdxs;
}

template <typename IdxTy, typename T>
thrust::device_vector<T> Compact(const thrust::device_vector<T> &d_srcs,
                                 const thrust::device_vector<IdxTy> &d_valids, IdxTy srcNum = 0) {
    auto d_compactedIdxs =
        CompactIndexes(srcNum == 0 ? static_cast<IdxTy>(d_srcs.size()) : srcNum, d_valids);

    thrust::device_vector<T> d_compacteds(d_compactedIdxs.size());
    thrust::for_each(thrust::make_counting_iterator(IdxTy(0)),
                     thrust::make_counting_iterator(static_cast<IdxTy>(d_compacteds.size())),
                     [compacteds = thrust::raw_pointer_cast(d_compacteds.data()),
                      compactedIdxs = thrust::raw_pointer_cast(d_compactedIdxs.data()),
                      srcs = thrust::raw_pointer_cast(d_srcs.data())] __device__(IdxTy cmpctIdx) {
                         compacteds[cmpctIdx] = srcs[compactedIdxs[cmpctIdx]];
                     });
    return d_compacteds;
}

} // namespace CUDA
} // namespace kouek

#endif // !KOUEK_CUDA_ALGORITHM_H
