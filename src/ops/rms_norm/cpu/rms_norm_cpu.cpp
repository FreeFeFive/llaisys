#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, size_t rows, size_t cols, float eps) {
    // For each row:
    // Y_i = (W_i * X_i) / sqrt(mean(X_i^2) + eps)
    // where mean(X_i^2) = (1/d) * sum(X_i[j]^2 for j in 0..d-1)
    for (size_t i = 0; i < rows; ++i) {
        const T *in_row = in + i * cols;
        T *out_row = out + i * cols;

        // Compute sum of squares
        float sum_sq = 0.0f;
        for (size_t j = 0; j < cols; ++j) {
            float val;
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                val = llaisys::utils::cast<float>(in_row[j]);
            } else {
                val = in_row[j];
            }
            sum_sq += val * val;
        }

        // Compute RMS normalization
        // mean = sum_sq / cols
        // rms = sqrt(mean + eps)
        float mean = sum_sq / static_cast<float>(cols);
        float rms = std::sqrt(mean + eps);
        float inv_rms = 1.0f / rms;

        // Apply normalization and weight
        for (size_t j = 0; j < cols; ++j) {
            float val;
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                val = llaisys::utils::cast<float>(in_row[j]);
            } else {
                val = in_row[j];
            }
            val = val * inv_rms;

            float w;
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                w = llaisys::utils::cast<float>(weight[j]);
            } else {
                w = weight[j];
            }
            val = val * w;

            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                out_row[j] = llaisys::utils::cast<T>(val);
            } else {
                out_row[j] = val;
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, size_t rows, size_t cols,
              llaisysDataType_t type, float eps) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                         reinterpret_cast<const float *>(weight), rows, cols, eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                         reinterpret_cast<const llaisys::bf16_t *>(weight), rows, cols, eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                         reinterpret_cast<const llaisys::fp16_t *>(weight), rows, cols, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
