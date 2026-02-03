#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                    size_t q_len, size_t kv_len, size_t n_heads, size_t n_kv_heads, size_t head_dim,
                    llaisysDataType_t type, float scale);
}
