#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cstring>

template <typename T>
void embedding_(T *out, const int64_t *index, const T *weight, size_t vocab_size, size_t embedding_dim,
                size_t seq_len) {
    for (size_t i = 0; i < seq_len; ++i) {
        int64_t idx = index[i];
        if (idx < 0 || idx >= static_cast<int64_t>(vocab_size)) {
            continue; // Skip invalid indices
        }
        std::memcpy(out + i * embedding_dim, weight + idx * embedding_dim, embedding_dim * sizeof(T));
    }
}

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, size_t vocab_size,
               size_t embedding_dim, llaisysDataType_t type, size_t seq_len) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return embedding_(reinterpret_cast<float *>(out), reinterpret_cast<const int64_t *>(index),
                          reinterpret_cast<const float *>(weight), vocab_size, embedding_dim, seq_len);
    case LLAISYS_DTYPE_BF16:
        return embedding_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const int64_t *>(index),
                          reinterpret_cast<const llaisys::bf16_t *>(weight), vocab_size, embedding_dim, seq_len);
    case LLAISYS_DTYPE_F16:
        return embedding_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const int64_t *>(index),
                          reinterpret_cast<const llaisys::fp16_t *>(weight), vocab_size, embedding_dim, seq_len);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
