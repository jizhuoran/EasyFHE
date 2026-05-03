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

namespace fhe {

/* kernel functions */

#define BARRET_PARAMS_0
#define BARRET_PARAMS_1 , const uint64_t* barret_mu

#define BARRET_ARGS_0
#define BARRET_ARGS_1 , barret_mu[l * 2], barret_mu[l * 2 + 1]

#define GENERATE_KERNEL(NAME, OP, B_ACCESS, HAS_BARRET)    \
  template <size_t NUM_CV>                                 \
  __global__ void NAME(                                    \
      const size_t N,                                      \
      const size_t LN_C,                                   \
      const size_t LN_A,                                   \
      const size_t LN_B,                                   \
      const size_t BLN_C,                                  \
      const size_t BLN_A,                                  \
      const size_t BLN_B,                                  \
      uint64_t* c,                                         \
      const uint64_t* a,                                   \
      const uint64_t* b,                                   \
      const uint64_t* mod BARRET_PARAMS_##HAS_BARRET) {    \
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;      \
    auto l = blockIdx.y;                                   \
    auto batch_id = blockIdx.z;                            \
    for (size_t i = 0; i < NUM_CV; i++) {                  \
      c[i * BLN_C + batch_id * LN_C + l * N + tid] =       \
          OP(a[i * BLN_A + batch_id * LN_A + l * N + tid], \
             B_ACCESS,                                     \
             mod[l] BARRET_ARGS_##HAS_BARRET);             \
    }                                                      \
  }

GENERATE_KERNEL(vadd_kernel, add_mod, b[i * BLN_B + batch_id * LN_B + l * N + tid], 0)
GENERATE_KERNEL(vsub_kernel, sub_mod, b[i * BLN_B + batch_id * LN_B + l * N + tid], 0)
GENERATE_KERNEL(vmul_kernel, mul_mod, b[i * BLN_B + batch_id * LN_B + l * N + tid], 1)

GENERATE_KERNEL(vadd_scalar_kernel, add_mod, b[l], 0)
GENERATE_KERNEL(vsub_scalar_kernel, sub_mod, b[l], 0)
GENERATE_KERNEL(vmul_scalar_kernel, mul_mod, b[l], 1)

GENERATE_KERNEL(vneg_kernel, neg_mod, b[l], 0)

GENERATE_KERNEL(vadd_pt_broadcast_kernel, add_mod, b[l * N + tid], 0)
GENERATE_KERNEL(vadd_pt_pairwise_kernel, add_mod, b[batch_id * LN_B + l * N + tid], 0)
GENERATE_KERNEL(vmul_pt_broadcast_kernel, mul_mod, b[l * N + tid], 1)
GENERATE_KERNEL(vmul_pt_pairwise_kernel, mul_mod, b[batch_id * LN_B + l * N + tid], 1)

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
      const size_t num_cv,                                               \
      const size_t batch,                                                \
      const size_t L_C,                                                  \
      const size_t L_A,                                                  \
      const size_t L_B,                                                  \
      const size_t N,                                                    \
      int64_t cur_limbs,                                                 \
      uint64_t* c,                                                       \
      const uint64_t* a,                                                 \
      const uint64_t* b,                                                 \
      const uint64_t* mod BARRET_PARAMS_##HAS_BARRET) {                  \
    auto LN_C = L_C * N;                                                 \
    auto LN_A = L_A * N;                                                 \
    auto LN_B = L_B * N;                                                 \
    auto BLN_C = batch * LN_C;                                           \
    auto BLN_A = batch * LN_A;                                           \
    auto BLN_B = batch * LN_B;                                           \
    if (num_cv == 1) {                                                   \
      fhe::NAME##_kernel<1>                                              \
          <<<dim3(num_blocks(N), cur_limbs, batch), dim3(BLOCK_SIZE)>>>( \
              N,                                                         \
              LN_C,                                                      \
              LN_A,                                                      \
              LN_B,                                                      \
              BLN_C,                                                     \
              BLN_A,                                                     \
              BLN_B,                                                     \
              c,                                                         \
              a,                                                         \
              b,                                                         \
              mod BARRET_ARGS_##HAS_BARRET);                             \
    } else if (num_cv == 2) {                                            \
      fhe::NAME##_kernel<2>                                              \
          <<<dim3(num_blocks(N), cur_limbs, batch), dim3(BLOCK_SIZE)>>>( \
              N,                                                         \
              LN_C,                                                      \
              LN_A,                                                      \
              LN_B,                                                      \
              BLN_C,                                                     \
              BLN_A,                                                     \
              BLN_B,                                                     \
              c,                                                         \
              a,                                                         \
              b,                                                         \
              mod BARRET_ARGS_##HAS_BARRET);                             \
    } else {                                                             \
      TORCH_INTERNAL_ASSERT(false, "Unsupported number of cvs");         \
    }                                                                    \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                                      \
  }

GENERATE_FUNCTION(vadd, 0)
GENERATE_FUNCTION(vsub, 0)
GENERATE_FUNCTION(vmul, 1)
GENERATE_FUNCTION(vadd_scalar, 0)
GENERATE_FUNCTION(vsub_scalar, 0)
GENERATE_FUNCTION(vmul_scalar, 1)
GENERATE_FUNCTION(vneg, 0)
GENERATE_FUNCTION(vadd_pt_broadcast, 0)
GENERATE_FUNCTION(vadd_pt_pairwise, 0)
GENERATE_FUNCTION(vmul_pt_broadcast, 1)
GENERATE_FUNCTION(vmul_pt_pairwise, 1)

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
    TORCH_INTERNAL_ASSERT(a.dim() == 4);                                       \
    auto num_cv = a.sizes()[0];                                                \
    auto batch = a.sizes()[1];                                                 \
    auto N = a.sizes()[3];                                                     \
    TORCH_INTERNAL_ASSERT(                                                     \
        (N == 1 << 6) || (N == 1 << 14) || (N == 1 << 15) || (N == 1 << 16) || \
        (N == 1 << 17) || (N == 1 << 18));                                     \
    NAME##_mod(                                                                \
        num_cv,                                                                \
        batch,                                                                 \
        c.sizes()[2],                                                          \
        a.sizes()[2],                                                          \
        b.sizes()[2],                                                          \
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
GENERATE_TEMPLATE(vadd_pt_broadcast, 0)
GENERATE_TEMPLATE(vadd_pt_pairwise, 0)
GENERATE_TEMPLATE(vmul_pt_broadcast, 1)
GENERATE_TEMPLATE(vmul_pt_pairwise, 1)

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

#define GENERATE_INTERFACE(NAME, HAS_BARRET)                                 \
  Tensor NAME##_mod_cuda(                                                    \
      const Tensor& a,                                                       \
      const Tensor& b,                                                       \
      const Tensor& mod BARRET_PARAMS_##HAS_BARRET,                          \
      int64_t cur_limbs) {                                                   \
    Tensor c = at::empty(                                                    \
        {a.sizes()[0], a.sizes()[1], cur_limbs, a.sizes()[3]}, a.options()); \
    v##NAME##_template(c, a, b, mod BARRET_ARGS_##HAS_BARRET, cur_limbs);    \
    return c;                                                                \
  }                                                                          \
                                                                             \
  Tensor& NAME##_mod_cuda_(                                                  \
      Tensor& self,                                                          \
      const Tensor& other,                                                   \
      const Tensor& mod BARRET_PARAMS_##HAS_BARRET,                          \
      int64_t cur_limbs) {                                                   \
    v##NAME##_template(                                                      \
        self, self, other, mod BARRET_ARGS_##HAS_BARRET, cur_limbs);         \
    return self;                                                             \
  }                                                                          \
                                                                             \
  Tensor& NAME##_mod_out_cuda(                                               \
      const Tensor& a,                                                       \
      const Tensor& b,                                                       \
      const Tensor& mod BARRET_PARAMS_##HAS_BARRET,                          \
      int64_t cur_limbs,                                                     \
      Tensor& c) {                                                           \
    v##NAME##_template(c, a, b, mod BARRET_ARGS_##HAS_BARRET, cur_limbs);    \
    return c;                                                                \
  }

GENERATE_INTERFACE(add, 0)
GENERATE_INTERFACE(sub, 0)
GENERATE_INTERFACE(mul, 1)
GENERATE_INTERFACE(add_scalar, 0)
GENERATE_INTERFACE(sub_scalar, 0)
GENERATE_INTERFACE(mul_scalar, 1)
GENERATE_INTERFACE(neg, 0)

#define GENERATE_PT_INTERFACE(NAME, HAS_BARRET)                              \
  Tensor NAME##_cuda(                                                        \
      const Tensor& a,                                                       \
      const Tensor& b,                                                       \
      const Tensor& mod BARRET_PARAMS_##HAS_BARRET,                          \
      int64_t cur_limbs) {                                                   \
    Tensor c = at::empty(                                                    \
        {a.sizes()[0], a.sizes()[1], cur_limbs, a.sizes()[3]}, a.options()); \
    v##NAME##_template(c, a, b, mod BARRET_ARGS_##HAS_BARRET, cur_limbs);    \
    return c;                                                                \
  }

GENERATE_PT_INTERFACE(add_pt_broadcast, 0)
GENERATE_PT_INTERFACE(add_pt_pairwise, 0)
GENERATE_PT_INTERFACE(mul_pt_broadcast, 1)
GENERATE_PT_INTERFACE(mul_pt_pairwise, 1)



#undef BARRET_PARAMS_0
#undef BARRET_PARAMS_1
#undef BARRET_ARGS_0
#undef BARRET_ARGS_1
#undef GENERATE_INTERFACE

} // namespace at::native
