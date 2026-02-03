#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    if (bias != nullptr) {
        CHECK_SAME_DEVICE(out, bias);
    }
    CHECK_ARGUMENT(in->ndim() == 2, "linear: in must be 2D");
    CHECK_ARGUMENT(weight->ndim() == 2, "linear: weight must be 2D");
    CHECK_ARGUMENT(out->ndim() == 2, "linear: out must be 2D");
    CHECK_ARGUMENT(in->shape()[0] == out->shape()[0], "linear: batch size mismatch");
    CHECK_ARGUMENT(weight->shape()[1] == in->shape()[1], "linear: in_features mismatch");
    CHECK_ARGUMENT(weight->shape()[0] == out->shape()[1], "linear: out_features mismatch");
    if (bias != nullptr) {
        CHECK_ARGUMENT(bias->ndim() == 1, "linear: bias must be 1D");
        CHECK_ARGUMENT(bias->shape()[0] == out->shape()[1], "linear: bias size mismatch");
    }
    CHECK_ARGUMENT(out->dtype() == in->dtype() && out->dtype() == weight->dtype(),
                   "linear: dtype mismatch");
    if (bias != nullptr) {
        CHECK_ARGUMENT(out->dtype() == bias->dtype(), "linear: bias dtype mismatch");
    }
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "Linear: output, input and weight tensors must be contiguous.");
    if (bias != nullptr) {
        ASSERT(bias->isContiguous(), "Linear: bias tensor must be contiguous.");
    }

    size_t seq_len = in->shape()[0];
    size_t in_features = in->shape()[1];
    size_t out_features = out->shape()[1];

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(), in->data(), weight->data(), bias != nullptr ? bias->data() : nullptr,
                           seq_len, in_features, out_features, out->dtype());
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(), bias != nullptr ? bias->data() : nullptr,
                           seq_len, in_features, out_features, out->dtype());
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
