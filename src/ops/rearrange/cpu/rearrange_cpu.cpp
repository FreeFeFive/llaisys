#include "rearrange_cpu.hpp"

#include "../../../utils.hpp"

#include <cstring>

template <typename T>
void rearrange_(T *out, const T *in, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &out_strides,
                const std::vector<ptrdiff_t> &in_strides, size_t ndim) {
    // Handle scalar case
    if (ndim == 0) {
        *out = *in;
        return;
    }

    // Calculate total number of elements
    size_t numel = 1;
    for (size_t i = 0; i < ndim; ++i) {
        numel *= shape[i];
    }

    // Create multi-dimensional index array
    std::vector<size_t> indices(ndim, 0);

    // Iterate over all elements
    for (size_t elem_idx = 0; elem_idx < numel; ++elem_idx) {
        // Calculate byte offsets
        ptrdiff_t out_offset = 0;
        ptrdiff_t in_offset = 0;
        for (size_t i = 0; i < ndim; ++i) {
            out_offset += indices[i] * out_strides[i];
            in_offset += indices[i] * in_strides[i];
        }

        // Copy element
        out[out_offset / static_cast<ptrdiff_t>(sizeof(T))] = in[in_offset / static_cast<ptrdiff_t>(sizeof(T))];

        // Increment indices (like nested loops)
        size_t dim = ndim - 1;
        while (dim < ndim) {
            indices[dim]++;
            if (indices[dim] < shape[dim]) {
                break;
            }
            indices[dim] = 0;
            dim--;
        }
    }
}

namespace llaisys::ops::cpu {
void rearrange(std::byte *out, const std::byte *in, const std::vector<size_t> &shape,
               const std::vector<ptrdiff_t> &out_strides, const std::vector<ptrdiff_t> &in_strides, size_t ndim,
               llaisysDataType_t type) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rearrange_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), shape, out_strides,
                          in_strides, ndim);
    case LLAISYS_DTYPE_BF16:
        return rearrange_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                          shape, out_strides, in_strides, ndim);
    case LLAISYS_DTYPE_F16:
        return rearrange_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                          shape, out_strides, in_strides, ndim);
    case LLAISYS_DTYPE_I64:
        return rearrange_(reinterpret_cast<int64_t *>(out), reinterpret_cast<const int64_t *>(in), shape,
                          out_strides, in_strides, ndim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
