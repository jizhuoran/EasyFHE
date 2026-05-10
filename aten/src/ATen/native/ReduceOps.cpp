#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/core/Tensor.h>
#include <ATen/native/ReduceOps.h>

namespace at::native {

// EasyFHE intentionally removes ordinary reduction kernels.
DEFINE_DISPATCH(sum_stub);
DEFINE_DISPATCH(nansum_stub);
DEFINE_DISPATCH(prod_stub);
DEFINE_DISPATCH(mean_stub);
DEFINE_DISPATCH(and_stub);
DEFINE_DISPATCH(or_stub);
DEFINE_DISPATCH(min_values_stub);
DEFINE_DISPATCH(max_values_stub);
DEFINE_DISPATCH(argmax_stub);
DEFINE_DISPATCH(argmin_stub);
DEFINE_DISPATCH(xor_sum_stub);
DEFINE_DISPATCH(std_var_stub);
DEFINE_DISPATCH(norm_kernel);
DEFINE_DISPATCH(norm_stub);
DEFINE_DISPATCH(powsum_stub);
DEFINE_DISPATCH(cumsum_stub);
DEFINE_DISPATCH(cumprod_stub);
DEFINE_DISPATCH(logcumsumexp_stub);
DEFINE_DISPATCH(aminmax_stub);
DEFINE_DISPATCH(aminmax_allreduce_stub);

} // namespace at::native
