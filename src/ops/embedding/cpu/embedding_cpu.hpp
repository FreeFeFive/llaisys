#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, size_t vocab_size,
               size_t embedding_dim, llaisysDataType_t type, size_t seq_len);
}
