#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

template <typename T>
void self_attention_(T *attn_val, const T *q, const T *k, const T *v, size_t q_len, size_t kv_len, size_t n_heads,
                     size_t n_kv_heads, size_t head_dim, float scale) {
    // Shape: q [q_len, n_heads, head_dim]
    //        k [kv_len, n_kv_heads, head_dim]
    //        v [kv_len, n_kv_heads, head_dim]
    //        attn_val [q_len, n_heads, head_dim]
    //
    // Computation:
    // 1. Compute attention scores: Q @ K^T * scale, shape [q_len, n_heads, kv_len]
    // 2. Apply causal mask (set future positions to -inf)
    // 3. Apply softmax
    // 4. Multiply by V to get output

    size_t heads_per_kv = n_heads / n_kv_heads; // For GQA support

    // Allocate temporary space for attention scores
    std::vector<float> attn_scores(q_len * n_heads * kv_len, 0.0f);

    // Step 1: Compute attention scores Q @ K^T * scale
    for (size_t i = 0; i < q_len; ++i) {
        for (size_t h = 0; h < n_heads; ++h) {
            size_t kv_h = h / heads_per_kv; // Map query head to KV head

            const T *q_vec = q + (i * n_heads + h) * head_dim;

            for (size_t j = 0; j < kv_len; ++j) {
                const T *k_vec = k + (j * n_kv_heads + kv_h) * head_dim;

                // Compute dot product
                float score = 0.0f;
                for (size_t d = 0; d < head_dim; ++d) {
                    float q_val, k_val;
                    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                        q_val = llaisys::utils::cast<float>(q_vec[d]);
                        k_val = llaisys::utils::cast<float>(k_vec[d]);
                    } else {
                        q_val = q_vec[d];
                        k_val = k_vec[d];
                    }
                    score += q_val * k_val;
                }
                score *= scale;
                attn_scores[(i * n_heads + h) * kv_len + j] = score;
            }
        }
    }

    // Step 2: Apply causal mask and softmax
    for (size_t i = 0; i < q_len; ++i) {
        for (size_t h = 0; h < n_heads; ++h) {
            float *scores = attn_scores.data() + (i * n_heads + h) * kv_len;

            // Apply causal mask: mask out positions j > i
            // In generative models, position i can only attend to positions <= i
            // But here we need to consider: can current position attend to future in KV cache?
            // The mask should prevent attending to positions after the current one
            for (size_t j = i + 1; j < kv_len; ++j) {
                scores[j] = -std::numeric_limits<float>::infinity();
            }

            // Compute softmax
            // Find max for numerical stability
            float max_score = -std::numeric_limits<float>::infinity();
            for (size_t j = 0; j < kv_len; ++j) {
                if (std::isfinite(scores[j])) {
                    max_score = std::max(max_score, scores[j]);
                }
            }

            // Compute exp and sum
            float sum_exp = 0.0f;
            for (size_t j = 0; j < kv_len; ++j) {
                if (std::isfinite(scores[j])) {
                    scores[j] = std::exp(scores[j] - max_score);
                    sum_exp += scores[j];
                } else {
                    scores[j] = 0.0f;
                }
            }

            // Normalize
            if (sum_exp > 0.0f) {
                for (size_t j = 0; j < kv_len; ++j) {
                    scores[j] /= sum_exp;
                }
            }
        }
    }

    // Step 3: Multiply attention weights by V
    for (size_t i = 0; i < q_len; ++i) {
        for (size_t h = 0; h < n_heads; ++h) {
            size_t kv_h = h / heads_per_kv;

            const float *weights = attn_scores.data() + (i * n_heads + h) * kv_len;
            T *output = attn_val + (i * n_heads + h) * head_dim;

            // output = sum_j(weights[j] * v[j])
            for (size_t d = 0; d < head_dim; ++d) {
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    output[d] = llaisys::utils::cast<T>(0.0f);
                } else {
                    output[d] = T(0);
                }
            }

            for (size_t j = 0; j < kv_len; ++j) {
                if (weights[j] > 0.0f) {
                    const T *v_vec = v + (j * n_kv_heads + kv_h) * head_dim;

                    for (size_t d = 0; d < head_dim; ++d) {
                        float v_val;
                        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                            v_val = llaisys::utils::cast<float>(v_vec[d]);
                            float out_val = llaisys::utils::cast<float>(output[d]);
                            out_val += weights[j] * v_val;
                            output[d] = llaisys::utils::cast<T>(out_val);
                        } else {
                            output[d] += weights[j] * v_vec[d];
                        }
                    }
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                    size_t q_len, size_t kv_len, size_t n_heads, size_t n_kv_heads, size_t head_dim,
                    llaisysDataType_t type, float scale) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(attn_val), reinterpret_cast<const float *>(q),
                               reinterpret_cast<const float *>(k), reinterpret_cast<const float *>(v), q_len,
                               kv_len, n_heads, n_kv_heads, head_dim, scale);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<llaisys::bf16_t *>(attn_val),
                               reinterpret_cast<const llaisys::bf16_t *>(q),
                               reinterpret_cast<const llaisys::bf16_t *>(k),
                               reinterpret_cast<const llaisys::bf16_t *>(v), q_len, kv_len, n_heads,
                               n_kv_heads, head_dim, scale);
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<llaisys::fp16_t *>(attn_val),
                               reinterpret_cast<const llaisys::fp16_t *>(q),
                               reinterpret_cast<const llaisys::fp16_t *>(k),
                               reinterpret_cast<const llaisys::fp16_t *>(v), q_len, kv_len, n_heads,
                               n_kv_heads, head_dim, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
