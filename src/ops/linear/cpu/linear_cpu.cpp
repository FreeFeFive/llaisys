#include "linear_cpu.hpp"

#include "../../../utils.hpp"

#include <cstring>

template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias, size_t seq_len, size_t in_features,
             size_t out_features) {
    // out = in @ weight^T + bias
    // out[i, j] = sum(in[i, k] * weight[j, k] for k in range(in_features)) + bias[j]
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < out_features; ++j) {
            float result = 0.0f;
            const T *in_row = in + i * in_features;
            const T *weight_row = weight + j * in_features;

            // Compute dot product
            for (size_t k = 0; k < in_features; ++k) {
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    result += llaisys::utils::cast<float>(in_row[k]) * llaisys::utils::cast<float>(weight_row[k]);
                } else {
                    result += in_row[k] * weight_row[k];
                }
            }

            // Add bias if present
            if (bias != nullptr) {
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    result += llaisys::utils::cast<float>(bias[j]);
                } else {
                    result += bias[j];
                }
            }

            // Store result
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                out[i * out_features + j] = llaisys::utils::cast<T>(result);
            } else {
                out[i * out_features + j] = result;
            }
        }
    }
}

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias, size_t seq_len,
            size_t in_features, size_t out_features, llaisysDataType_t type) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                       reinterpret_cast<const float *>(weight), reinterpret_cast<const float *>(bias), seq_len,
                       in_features, out_features);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                       reinterpret_cast<const llaisys::bf16_t *>(weight), reinterpret_cast<const llaisys::bf16_t *>(bias),
                       seq_len, in_features, out_features);
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                       reinterpret_cast<const llaisys::fp16_t *>(weight), reinterpret_cast<const llaisys::fp16_t *>(bias),
                       seq_len, in_features, out_features);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
