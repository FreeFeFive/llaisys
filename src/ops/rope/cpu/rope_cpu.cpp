#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids, size_t seq_len, size_t n_heads, size_t head_dim,
           float theta) {
    // out shape: [seq_len, n_heads, head_dim]
    // in shape: [seq_len, n_heads, head_dim]
    // pos_ids shape: [seq_len]
    // head_dim must be even
    size_t half_dim = head_dim / 2;

    for (size_t s = 0; s < seq_len; ++s) {
        float pos = static_cast<float>(pos_ids[s]);

        for (size_t h = 0; h < n_heads; ++h) {
            // Get pointers to current sequence and head
            const T *in_head = in + (s * n_heads + h) * head_dim;
            T *out_head = out + (s * n_heads + h) * head_dim;

            // Apply RoPE to each pair (a, b)
            for (size_t j = 0; j < half_dim; ++j) {
                // Compute frequency: theta^(2j/d)
                float freq_exp = 2.0f * j / static_cast<float>(head_dim);
                float freq = pos / std::pow(theta, freq_exp);

                // Compute sin and cos
                float cos_freq = std::cos(freq);
                float sin_freq = std::sin(freq);

                // Get a and b values
                float a, b;
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    a = llaisys::utils::cast<float>(in_head[j]);
                    b = llaisys::utils::cast<float>(in_head[half_dim + j]);
                } else {
                    a = in_head[j];
                    b = in_head[half_dim + j];
                }

                // Compute rotated values
                // a' = a * cos - b * sin
                // b' = b * cos + a * sin
                float a_rot = a * cos_freq - b * sin_freq;
                float b_rot = b * cos_freq + a * sin_freq;

                // Store results
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    out_head[j] = llaisys::utils::cast<T>(a_rot);
                    out_head[half_dim + j] = llaisys::utils::cast<T>(b_rot);
                } else {
                    out_head[j] = a_rot;
                    out_head[half_dim + j] = b_rot;
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, size_t seq_len, size_t n_heads,
          size_t head_dim, llaisysDataType_t type, float theta) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                     reinterpret_cast<const int64_t *>(pos_ids), seq_len, n_heads, head_dim, theta);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                     reinterpret_cast<const int64_t *>(pos_ids), seq_len, n_heads, head_dim, theta);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                     reinterpret_cast<const int64_t *>(pos_ids), seq_len, n_heads, head_dim, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
