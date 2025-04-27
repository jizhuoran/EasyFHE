#include <ATen/Dispatch_v2.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/fhe/cpu/arithmetic.h>
#include <ATen/ops/copy.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>
#include <omp.h>
#include <cassert>
#pragma clang diagnostic ignored "-Wmissing-prototypes"

#define WORK_PER_THREAD (1)
#define WARP_SIZE (32)
#define NUM_WARPS (1)
#define BLOCK_SIZE (WARP_SIZE * NUM_WARPS)
#define WORK_PER_BLOCK (WORK_PER_THREAD * BLOCK_SIZE)

#define num_blocks(n) ((n + WORK_PER_BLOCK - 1) / WORK_PER_BLOCK)

namespace fhe {

/* kernel functions */

#define BARRET_PARAMS_0
#define BARRET_PARAMS_1 , const uint64_t* barret_mu

#define BARRET_ARGS_0
#define BARRET_ARGS_1 , barret_mu[l * 2], barret_mu[l * 2 + 1]

#define GENERATE_KERNEL(NAME, OP, B_ACCESS, HAS_BARRET)                         \
  void NAME(                                                                    \
      const size_t L,                                                           \
      const size_t N,                                                           \
      uint64_t* c,                                                              \
      const uint64_t* a,                                                        \
      const uint64_t* b,                                                        \
      const uint64_t* mod BARRET_PARAMS_##HAS_BARRET) {                         \
    const int max_threads = omp_get_max_threads();                              \
    omp_set_num_threads(max_threads);                                           \
    _Pragma("omp parallel for schedule(static) num_threads(max_threads)") for ( \
        size_t l = 0; l < L; l++) {                                             \
      for (size_t i = 0; i < N; i++) {                                          \
        c[l * N + i] =                                                          \
            OP(a[l * N + i], B_ACCESS, mod[l] BARRET_ARGS_##HAS_BARRET);        \
      }                                                                         \
    }                                                                           \
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
#define BARRET_PARAMS_1 , const Tensor& barret_mu

#define BARRET_ARGS_0
#define BARRET_ARGS_1 , barret_mu.data_ptr<uint64_t>()

#define GENERATE_KERNEL(NAME, HAS_BARRET)                   \
  static void NAME##_template(                              \
      Tensor& c,                                            \
      const Tensor& a,                                      \
      const Tensor& b,                                      \
      const Tensor& mod BARRET_PARAMS_##HAS_BARRET,         \
      int64_t cur_limbs) {                                  \
    TORCH_INTERNAL_ASSERT(a.dim() == 2);                    \
    auto N = static_cast<int>(a.sizes()[1]);                \
    fhe::NAME##_kernel(                                     \
        cur_limbs,                                          \
        N,                                                  \
        c.mutable_data_ptr<uint64_t>(),                     \
        a.data_ptr<uint64_t>(),                             \
        b.data_ptr<uint64_t>(),                             \
        mod.data_ptr<uint64_t>() BARRET_ARGS_##HAS_BARRET); \
  }

GENERATE_KERNEL(vadd, 0)
GENERATE_KERNEL(vsub, 0)
GENERATE_KERNEL(vmul, 1)
GENERATE_KERNEL(vadd_scalar, 0)
GENERATE_KERNEL(vsub_scalar, 0)
GENERATE_KERNEL(vmul_scalar, 1)
GENERATE_KERNEL(vneg, 0)

#undef BARRET_PARAMS_0
#undef BARRET_PARAMS_1
#undef BARRET_ARGS_0
#undef BARRET_ARGS_1
#undef GENERATE_KERNEL

/* functions */

#define BARRET_PARAMS_0
#define BARRET_PARAMS_1 , const Tensor& barret_mu

#define BARRET_ARGS_0
#define BARRET_ARGS_1 , barret_mu

#define GENERATE_FUNCTION(NAME, HAS_BARRET)                               \
  Tensor NAME##_mod_cpu(                                                  \
      const Tensor& a,                                                    \
      const Tensor& b,                                                    \
      const Tensor& mod BARRET_PARAMS_##HAS_BARRET,                       \
      int64_t cur_limbs) {                                                \
    Tensor c = at::empty_like(a);                                         \
    v##NAME##_template(c, a, b, mod BARRET_ARGS_##HAS_BARRET, cur_limbs); \
    return c;                                                             \
  }                                                                       \
                                                                          \
  Tensor& NAME##_mod_cpu_(                                                \
      Tensor& self,                                                       \
      const Tensor& other,                                                \
      const Tensor& mod BARRET_PARAMS_##HAS_BARRET,                       \
      int64_t cur_limbs) {                                                \
    v##NAME##_template(                                                   \
        self, self, other, mod BARRET_ARGS_##HAS_BARRET, cur_limbs);      \
    return self;                                                          \
  }                                                                       \
                                                                          \
  Tensor& NAME##_mod_out_cpu(                                             \
      const Tensor& a,                                                    \
      const Tensor& b,                                                    \
      const Tensor& mod BARRET_PARAMS_##HAS_BARRET,                       \
      int64_t cur_limbs,                                                  \
      Tensor& c) {                                                        \
    v##NAME##_template(c, a, b, mod BARRET_ARGS_##HAS_BARRET, cur_limbs); \
    return c;                                                             \
  }

GENERATE_FUNCTION(add, 0)
GENERATE_FUNCTION(sub, 0)
GENERATE_FUNCTION(mul, 1)
GENERATE_FUNCTION(add_scalar, 0)
GENERATE_FUNCTION(sub_scalar, 0)
GENERATE_FUNCTION(mul_scalar, 1)
GENERATE_FUNCTION(neg, 0)

#undef BARRET_PARAMS_0
#undef BARRET_PARAMS_1
#undef BARRET_ARGS_0
#undef BARRET_ARGS_1
#undef GENERATE_FUNCTION

} // namespace at::native