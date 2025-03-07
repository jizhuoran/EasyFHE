#include <ATen/Dispatch_v2.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/thread_constants.h>
#include <ATen/ops/copy.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>
#include <cudaFHE.h>
#include <cassert>

#pragma clang diagnostic ignored "-Wmissing-prototypes"

namespace at::native {

#define BARRET_PARAMS_0
#define BARRET_PARAMS_1 , const Tensor& barret_mu

#define BARRET_ARGS_0
#define BARRET_ARGS_1 , barret_mu.data_ptr<uint64_t>()

#define GENERATE_INTERFACE(NAME, HAS_BARRET)               \
  Tensor NAME##_mod_cuda(                                  \
      const Tensor& a,                                     \
      const Tensor& b,                                     \
      const Tensor& mod BARRET_PARAMS_##HAS_BARRET,        \
      int64_t cur_limbs) {                                 \
    Tensor c = at::empty_like(a);                          \
    auto N = static_cast<int>(a.sizes()[1]);               \
    cudaFHE::v##NAME##_template(                           \
        c.mutable_data_ptr<uint64_t>(),                    \
        a.data_ptr<uint64_t>(),                            \
        b.data_ptr<uint64_t>(),                            \
        mod.data_ptr<uint64_t>() BARRET_ARGS_##HAS_BARRET, \
        N,                                                 \
        cur_limbs);                                        \
    return c;                                              \
  }                                                        \
                                                           \
  Tensor& NAME##_mod_cuda_(                                \
      Tensor& self,                                        \
      const Tensor& other,                                 \
      const Tensor& mod BARRET_PARAMS_##HAS_BARRET,        \
      int64_t cur_limbs) {                                 \
    auto N = static_cast<int>(self.sizes()[1]);            \
    cudaFHE::v##NAME##_template(                           \
        self.mutable_data_ptr<uint64_t>(),                 \
        self.data_ptr<uint64_t>(),                         \
        other.data_ptr<uint64_t>(),                        \
        mod.data_ptr<uint64_t>() BARRET_ARGS_##HAS_BARRET, \
        N,                                                 \
        cur_limbs);                                        \
    return self;                                           \
  }                                                        \
                                                           \
  Tensor& NAME##_mod_out_cuda(                             \
      const Tensor& a,                                     \
      const Tensor& b,                                     \
      const Tensor& mod BARRET_PARAMS_##HAS_BARRET,        \
      int64_t cur_limbs,                                   \
      Tensor& c) {                                         \
    auto N = static_cast<int>(a.sizes()[1]);               \
    cudaFHE::v##NAME##_template(                           \
        c.mutable_data_ptr<uint64_t>(),                    \
        a.data_ptr<uint64_t>(),                            \
        b.data_ptr<uint64_t>(),                            \
        mod.data_ptr<uint64_t>() BARRET_ARGS_##HAS_BARRET, \
        N,                                                 \
        cur_limbs);                                        \
    return c;                                              \
  }

GENERATE_INTERFACE(add, 0)
GENERATE_INTERFACE(sub, 0)
GENERATE_INTERFACE(mul, 1)
GENERATE_INTERFACE(add_scalar, 0)
GENERATE_INTERFACE(sub_scalar, 0)
GENERATE_INTERFACE(mul_scalar, 1)
GENERATE_INTERFACE(neg, 0)

#undef BARRET_PARAMS_0
#undef BARRET_PARAMS_1
#undef BARRET_ARGS_0
#undef BARRET_ARGS_1
#undef GENERATE_INTERFACE

} // namespace at::native