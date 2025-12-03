#include <ATen/Dispatch_v2.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/fhe/cpu/arithmetic.h>
#include <ATen/ops/copy.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>
#include <immintrin.h>
#include <omp.h>
#include <cassert>
#include <iostream>
#pragma clang diagnostic ignored "-Wmissing-prototypes"

#define WORK_PER_THREAD (1)
#define WARP_SIZE (32)
#define NUM_WARPS (1)
#define BLOCK_SIZE (WARP_SIZE * NUM_WARPS)
#define WORK_PER_BLOCK (WORK_PER_THREAD * BLOCK_SIZE)

#define num_blocks(n) ((n + WORK_PER_BLOCK - 1) / WORK_PER_BLOCK)

namespace fhe {
#ifdef USE_AVX512
#define BARRET_PARAMS_0
#define BARRET_PARAMS_1 , const uint64_t* barret_mu

#define BARRET_ARGS_VEC_0
#define BARRET_ARGS_VEC_1 \
  , _mm512_set1_epi64(barret_mu[l * 2]), _mm512_set1_epi64(barret_mu[l * 2 + 1])

#define BARRET_ARGS_TAIL_0
#define BARRET_ARGS_TAIL_1 , barret_mu[l * 2], barret_mu[l * 2 + 1]

#define GENERATE_KERNEL_AVX(                                                     \
    NAME, OP_VEC, OP_SCALAR, B_IS_SCALAR, HAS_BARRET, B_ACCESS)                  \
  void NAME(                                                                     \
      const size_t L,                                                            \
      const size_t N,                                                            \
      uint64_t* c,                                                               \
      const uint64_t* a,                                                         \
      const uint64_t* b,                                                         \
      const uint64_t* mod BARRET_PARAMS_##HAS_BARRET) {                          \
    const int max_threads = omp_get_max_threads();                               \
    _Pragma("omp parallel for  schedule(static) num_threads(max_threads)") for ( \
        size_t l = 0; l < L; l++) {                                              \
      size_t i = 0;                                                              \
      for (; i + 8 <= N; i += 8) {                                               \
        __m512i vec_a = _mm512_loadu_si512(&a[l * N + i]);                       \
        __m512i vec_b = B_IS_SCALAR ? _mm512_set1_epi64(B_ACCESS)                \
                                    : _mm512_loadu_si512(&B_ACCESS);             \
        __m512i vec_mod = _mm512_set1_epi64(mod[l]);                             \
        __m512i vec_res =                                                        \
            OP_VEC(vec_a, vec_b, vec_mod BARRET_ARGS_VEC_##HAS_BARRET);          \
        _mm512_storeu_si512(&c[l * N + i], vec_res);                             \
      }                                                                          \
      for (; i < N; i++) {                                                       \
        c[l * N + i] = OP_SCALAR(                                                \
            a[l * N + i], B_ACCESS, mod[l] BARRET_ARGS_TAIL_##HAS_BARRET);       \
      }                                                                          \
    }                                                                            \
  }

GENERATE_KERNEL_AVX(vadd_kernel, add_mod_avx512, add_mod, 0, 0, b[l * N + i])
GENERATE_KERNEL_AVX(vsub_kernel, sub_mod_avx512, sub_mod, 0, 0, b[l * N + i])
GENERATE_KERNEL_AVX(vmul_kernel, mul_mod_avx512, mul_mod, 0, 1, b[l * N + i])
GENERATE_KERNEL_AVX(vadd_scalar_kernel, add_mod_avx512, add_mod, 1, 0, b[l])
GENERATE_KERNEL_AVX(vsub_scalar_kernel, sub_mod_avx512, sub_mod, 1, 0, b[l])
GENERATE_KERNEL_AVX(vmul_scalar_kernel, mul_mod_avx512, mul_mod, 1, 1, b[l])

GENERATE_KERNEL_AVX(vneg_kernel, neg_mod_avx512, neg_mod, 1, 0, b[l])
#undef BARRET_PARAMS_0
#undef BARRET_PARAMS_1
#undef BARRET_ARGS_VEC_0
#undef BARRET_ARGS_VEC_1
#undef BARRET_ARGS_TAIL_0
#undef BARRET_ARGS_TAIL_1
#undef VLOAD_B_0
#undef VLOAD_B_1
#undef TAIL_B_0
#undef TAIL_B_1
#undef GENERATE_KERNEL_AVX
#undef GENERATE_KERNEL
#else
/* kernel functions */

#define BARRET_PARAMS_0
#define BARRET_PARAMS_1 , const uint64_t* barret_mu

#define BARRET_ARGS_0
#define BARRET_ARGS_1 , barret_mu[l * 2], barret_mu[l * 2 + 1]

#define GENERATE_KERNEL(NAME, OP, B_ACCESS, HAS_BARRET)                                     \
  void NAME(                                                                                \
      const size_t L,                                                                       \
      const size_t N,                                                                       \
      uint64_t* c,                                                                          \
      const uint64_t* a,                                                                    \
      const uint64_t* b,                                                                    \
      const uint64_t* mod BARRET_PARAMS_##HAS_BARRET) {                                     \
    const int max_threads = omp_get_max_threads();                                          \
    _Pragma("omp parallel for collapse(2) schedule(static) num_threads(max_threads)") for ( \
        size_t l = 0; l < L; l++) {                                                         \
      for (size_t i = 0; i < N; i++) {                                                      \
        c[l * N + i] =                                                                      \
            OP(a[l * N + i], B_ACCESS, mod[l] BARRET_ARGS_##HAS_BARRET);                    \
      }                                                                                     \
    }                                                                                       \
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
#endif
} // namespace fhe

namespace at::native {

/* kernel launchers */

#define BARRET_PARAMS_0
#define BARRET_PARAMS_1 , const uint64_t* barret_mu

#define BARRET_ARGS_0
#define BARRET_ARGS_1 , barret_mu

#define GENERATE_FUNCTION(NAME, HAS_BARRET)                          \
  void NAME##_mod(                                                   \
      const size_t N,                                                \
      int64_t l,                                                     \
      uint64_t* c,                                                   \
      const uint64_t* a,                                             \
      const uint64_t* b,                                             \
      const uint64_t* mod BARRET_PARAMS_##HAS_BARRET) {              \
    fhe::NAME##_kernel(l, N, c, a, b, mod BARRET_ARGS_##HAS_BARRET); \
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

#define GENERATE_TEMPLATE(NAME, HAS_BARRET)                 \
  static void NAME##_template(                              \
      Tensor& c,                                            \
      const Tensor& a,                                      \
      const Tensor& b,                                      \
      const Tensor& mod BARRET_PARAMS_##HAS_BARRET,         \
      int64_t cur_limbs) {                                  \
    TORCH_INTERNAL_ASSERT(a.dim() == 2);                    \
    auto N = static_cast<int>(a.sizes()[1]);                \
    NAME##_mod(                                             \
        N,                                                  \
        cur_limbs,                                          \
        c.mutable_data_ptr<uint64_t>(),                     \
        a.data_ptr<uint64_t>(),                             \
        b.data_ptr<uint64_t>(),                             \
        mod.data_ptr<uint64_t>() BARRET_ARGS_##HAS_BARRET); \
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
