#include <ATen/Dispatch_v2.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/thread_constants.h>
#include <ATen/native/fhe/cuda/arithmetic.h>
#include <ATen/native/fhe/cuda/Utils.cuh>
#include <ATen/ops/copy.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>
#include <cassert>


#pragma clang diagnostic ignored "-Wmissing-prototypes"

#define WORK_PER_THREAD (1)
#define WARP_SIZE (32)
#define NUM_WARPS (8)
#define BLOCK_SIZE (WARP_SIZE * NUM_WARPS)
#define WORK_PER_BLOCK (WORK_PER_THREAD * BLOCK_SIZE)

#define num_blocks(n) ((n + WORK_PER_BLOCK - 1) / WORK_PER_BLOCK)

namespace fhe {

/* kernel functions */

#define BARRET_PARAMS_0
#define BARRET_PARAMS_1 , const uint64_t* barret_mu

#define BARRET_ARGS_0
#define BARRET_ARGS_1 , barret_mu[l * 2], barret_mu[l * 2 + 1]

#define GENERATE_KERNEL(NAME, OP, B_ACCESS, HAS_BARRET)              \
  __global__ void NAME(                                              \
      const size_t N,                                                \
      uint64_t* c,                                                   \
      const uint64_t* a,                                             \
      const uint64_t* b,                                             \
      const uint64_t* mod BARRET_PARAMS_##HAS_BARRET) {              \
    auto i = blockIdx.x * blockDim.x + threadIdx.x;                  \
    auto l = blockIdx.y;                                             \
    c[l * N + i] =                                                   \
        OP(a[l * N + i], B_ACCESS, mod[l] BARRET_ARGS_##HAS_BARRET); \
  }

GENERATE_KERNEL(vadd_kernel, add_mod, b[l * N + i], 0)
GENERATE_KERNEL(vsub_kernel, sub_mod, b[l * N + i], 0)
GENERATE_KERNEL(vmul_kernel, mul_mod, b[l * N + i], 1)

GENERATE_KERNEL(vadd_scalar_kernel, add_mod, b[l], 0)
GENERATE_KERNEL(vsub_scalar_kernel, sub_mod, b[l], 0)
GENERATE_KERNEL(vmul_scalar_kernel, mul_mod, b[l], 1)

GENERATE_KERNEL(vneg_kernel, neg_mod, b[l], 0)

#undef BARRET_PARAMS_0
#undef BARRET_PARAMS_1
#undef BARRET_ARGS_0
#undef BARRET_ARGS_1
#undef GENERATE_KERNEL

} // namespace fhe

namespace at::native {

/* kernel launchers */

#define BARRET_PARAMS_0
#define BARRET_PARAMS_1 , const uint64_t* barret_mu

#define BARRET_ARGS_0
#define BARRET_ARGS_1 , barret_mu

#define GENERATE_FUNCTION(NAME, HAS_BARRET)                              \
  void NAME##_mod(                                                       \
      const size_t N,                                                    \
      int64_t l,                                                         \
      uint64_t* c,                                                       \
      const uint64_t* a,                                                 \
      const uint64_t* b,                                                 \
      const uint64_t* mod BARRET_PARAMS_##HAS_BARRET) {                  \
    fhe::NAME##_kernel<<<dim3(num_blocks(N), l), dim3(BLOCK_SIZE, 1)>>>( \
        N, c, a, b, mod BARRET_ARGS_##HAS_BARRET);                       \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                                      \
  }

GENERATE_FUNCTION(vadd, 0)
GENERATE_FUNCTION(vsub, 0)
GENERATE_FUNCTION(vmul, 1)
GENERATE_FUNCTION(vadd_scalar, 0)
GENERATE_FUNCTION(vsub_scalar, 0)
GENERATE_FUNCTION(vmul_scalar, 1)
GENERATE_FUNCTION(vneg, 0)

#undef BARRET_PARAMS_0
#undef BARRET_PARAMS_1
#undef BARRET_ARGS_0
#undef BARRET_ARGS_1
#undef GENERATE_FUNCTION

/* templates */

#define BARRET_PARAMS_0
#define BARRET_PARAMS_1 , const Tensor& barret_mu

#define BARRET_ARGS_0
#define BARRET_ARGS_1 , barret_mu.data_ptr<uint64_t>()

#define GENERATE_TEMPLATE(NAME, HAS_BARRET)                                    \
  static void NAME##_template(                                                 \
      Tensor& c,                                                               \
      const Tensor& a,                                                         \
      const Tensor& b,                                                         \
      const Tensor& mod BARRET_PARAMS_##HAS_BARRET,                            \
      int64_t cur_limbs) {                                                     \
    TORCH_INTERNAL_ASSERT(a.dim() == 2);                                       \
    auto N = static_cast<int>(a.sizes()[1]);                                   \
    TORCH_INTERNAL_ASSERT(                                                     \
        (N == 1 << 6) || (N == 1 << 14) || (N == 1 << 15) || (N == 1 << 16) || \
        (N == 1 << 17) || (N == 1 << 18));                                     \
    NAME##_mod(                                                                \
        N,                                                                     \
        cur_limbs,                                                             \
        c.mutable_data_ptr<uint64_t>(),                                        \
        a.data_ptr<uint64_t>(),                                                \
        b.data_ptr<uint64_t>(),                                                \
        mod.data_ptr<uint64_t>() BARRET_ARGS_##HAS_BARRET);                    \
  }

GENERATE_TEMPLATE(vadd, 0)
GENERATE_TEMPLATE(vsub, 0)
GENERATE_TEMPLATE(vmul, 1)
GENERATE_TEMPLATE(vadd_scalar, 0)
GENERATE_TEMPLATE(vsub_scalar, 0)
GENERATE_TEMPLATE(vmul_scalar, 1)
GENERATE_TEMPLATE(vneg, 0)

#undef BARRET_PARAMS_0
#undef BARRET_PARAMS_1
#undef BARRET_ARGS_0
#undef BARRET_ARGS_1
#undef GENERATE_TEMPLATE

/* interface */

#define BARRET_PARAMS_0
#define BARRET_PARAMS_1 , const Tensor& barret_mu

#define BARRET_ARGS_0
#define BARRET_ARGS_1 , barret_mu

#define GENERATE_INTERFACE(NAME, HAS_BARRET)                              \
  Tensor NAME##_mod_cuda(                                                 \
      const Tensor& a,                                                    \
      const Tensor& b,                                                    \
      const Tensor& mod BARRET_PARAMS_##HAS_BARRET,                       \
      int64_t cur_limbs) {                                                \
    Tensor c = at::empty_like(a);                                         \
    v##NAME##_template(c, a, b, mod BARRET_ARGS_##HAS_BARRET, cur_limbs); \
    return c;                                                             \
  }                                                                       \
                                                                          \
  Tensor& NAME##_mod_cuda_(                                               \
      Tensor& self,                                                       \
      const Tensor& other,                                                \
      const Tensor& mod BARRET_PARAMS_##HAS_BARRET,                       \
      int64_t cur_limbs) {                                                \
    v##NAME##_template(                                                   \
        self, self, other, mod BARRET_ARGS_##HAS_BARRET, cur_limbs);      \
    return self;                                                          \
  }                                                                       \
                                                                          \
  Tensor& NAME##_mod_out_cuda(                                            \
      const Tensor& a,                                                    \
      const Tensor& b,                                                    \
      const Tensor& mod BARRET_PARAMS_##HAS_BARRET,                       \
      int64_t cur_limbs,                                                  \
      Tensor& c) {                                                        \
    v##NAME##_template(c, a, b, mod BARRET_ARGS_##HAS_BARRET, cur_limbs); \
    return c;                                                             \
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
