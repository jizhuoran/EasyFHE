#include <ATen/native/Activation.h>

namespace at::native {

DEFINE_DISPATCH(elu_stub);
DEFINE_DISPATCH(elu_backward_stub);
DEFINE_DISPATCH(softplus_stub);
DEFINE_DISPATCH(softplus_backward_stub);
DEFINE_DISPATCH(log_sigmoid_cpu_stub);
DEFINE_DISPATCH(log_sigmoid_backward_stub);
DEFINE_DISPATCH(threshold_stub);
DEFINE_DISPATCH(GeluKernel);
DEFINE_DISPATCH(GeluBackwardKernel);
DEFINE_DISPATCH(hardtanh_backward_stub);
DEFINE_DISPATCH(hardsigmoid_stub);
DEFINE_DISPATCH(hardsigmoid_backward_stub);
DEFINE_DISPATCH(hardswish_stub);
DEFINE_DISPATCH(hardswish_backward_stub);
DEFINE_DISPATCH(hardshrink_stub);
DEFINE_DISPATCH(softshrink_stub);
DEFINE_DISPATCH(shrink_backward_stub);
DEFINE_DISPATCH(leaky_relu_stub);
DEFINE_DISPATCH(leaky_relu_backward_stub);
DEFINE_DISPATCH(glu_stub);
DEFINE_DISPATCH(glu_backward_stub);
DEFINE_DISPATCH(glu_jvp_stub);
DEFINE_DISPATCH(silu_stub);
DEFINE_DISPATCH(silu_backward_stub);
DEFINE_DISPATCH(mish_stub);
DEFINE_DISPATCH(mish_backward_stub);
DEFINE_DISPATCH(prelu_stub);
DEFINE_DISPATCH(prelu_backward_stub);

} // namespace at::native
