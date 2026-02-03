#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_ARGUMENT(q->ndim() == 3, "self_attention: q must be 3D");
    CHECK_ARGUMENT(k->ndim() == 3, "self_attention: k must be 3D");
    CHECK_ARGUMENT(v->ndim() == 3, "self_attention: v must be 3D");
    CHECK_ARGUMENT(attn_val->ndim() == 3, "self_attention: attn_val must be 3D");

    size_t q_len = q->shape()[0];
    size_t n_heads = q->shape()[1];
    size_t head_dim = q->shape()[2];
    size_t kv_len = k->shape()[0];
    size_t n_kv_heads = k->shape()[1];

    CHECK_ARGUMENT(q->shape()[2] == k->shape()[2], "self_attention: q and k head_dim must match");
    CHECK_ARGUMENT(k->shape()[2] == v->shape()[2], "self_attention: k and v head_dim must match");
    CHECK_ARGUMENT(k->shape()[0] == v->shape()[0], "self_attention: k and v seq_len must match");
    CHECK_ARGUMENT(k->shape()[1] == v->shape()[1], "self_attention: k and v n_heads must match");
    CHECK_ARGUMENT(attn_val->shape()[0] == q_len && attn_val->shape()[1] == n_heads && attn_val->shape()[2] == head_dim,
                   "self_attention: attn_val shape mismatch");
    CHECK_ARGUMENT(q->dtype() == k->dtype() && q->dtype() == v->dtype() && q->dtype() == attn_val->dtype(),
                   "self_attention: dtype mismatch");
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),
           "SelfAttention: all tensors must be contiguous.");

    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), q_len, kv_len, n_heads,
                                   n_kv_heads, head_dim, attn_val->dtype(), scale);
    }

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), q_len, kv_len, n_heads,
                                   n_kv_heads, head_dim, attn_val->dtype(), scale);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
