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

namespace fhe {

/* simple kernels for 2D [L, N] buffers */

#define BARRET_PARAMS_0
#define BARRET_PARAMS_1 , const uint64_t* barret_mu

#define BARRET_ARGS_0
#define BARRET_ARGS_1 , barret_mu[l * 2], barret_mu[l * 2 + 1]

#ifdef USE_AVX512
#define BARRET_ARGS_VEC_0
#define BARRET_ARGS_VEC_1 \
  , _mm512_set1_epi64(barret_mu[l * 2]), _mm512_set1_epi64(barret_mu[l * 2 + 1])
#define BARRET_ARGS_SCALAR_0
#define BARRET_ARGS_SCALAR_1 , barret_mu[l * 2], barret_mu[l * 2 + 1]
#define GENERATE_SIMPLE_KERNEL(NAME, OP_VEC, OP_SCALAR, B_IS_SCALAR, B_ACCESS, HAS_BARRET) \
  static void NAME(                                                             \
      const size_t L,                                                           \
      const size_t N,                                                           \
      uint64_t* c,                                                              \
      const uint64_t* a,                                                        \
      const uint64_t* b,                                                        \
      const uint64_t* mod BARRET_PARAMS_##HAS_BARRET) {                         \
    const int max_threads = omp_get_max_threads();                              \
    _Pragma("omp parallel for schedule(static) num_threads(max_threads)")        \
    for (size_t l = 0; l < L; ++l) {                                            \
      size_t i = 0;                                                             \
      const __m512i vec_mod = _mm512_set1_epi64(mod[l]);                       \
      for (; i + 8 <= N; i += 8) {                                              \
        const __m512i vec_a = _mm512_loadu_si512(&a[l * N + i]);               \
        const __m512i vec_b = (B_IS_SCALAR) ? _mm512_set1_epi64(B_ACCESS)      \
                                            : _mm512_loadu_si512(&B_ACCESS);    \
        const __m512i vec_r = OP_VEC(vec_a, vec_b, vec_mod BARRET_ARGS_VEC_##HAS_BARRET); \
        _mm512_storeu_si512(&c[l * N + i], vec_r);                              \
      }                                                                         \
      for (; i < N; ++i) {                                                       \
        c[l * N + i] =                                                           \
            OP_SCALAR(a[l * N + i], B_ACCESS, mod[l] BARRET_ARGS_SCALAR_##HAS_BARRET); \
      }                                                                         \
    }                                                                           \
  }
GENERATE_SIMPLE_KERNEL(vadd_simple_kernel, add_mod_avx512, add_mod, 0, b[l * N + i], 0)
GENERATE_SIMPLE_KERNEL(vsub_simple_kernel, sub_mod_avx512, sub_mod, 0, b[l * N + i], 0)
GENERATE_SIMPLE_KERNEL(vmul_simple_kernel, mul_mod_avx512, mul_mod, 0, b[l * N + i], 1)
GENERATE_SIMPLE_KERNEL(vadd_scalar_simple_kernel, add_mod_avx512, add_mod, 1, b[l], 0)
GENERATE_SIMPLE_KERNEL(vsub_scalar_simple_kernel, sub_mod_avx512, sub_mod, 1, b[l], 0)
GENERATE_SIMPLE_KERNEL(vmul_scalar_simple_kernel, mul_mod_avx512, mul_mod, 1, b[l], 1)
GENERATE_SIMPLE_KERNEL(vneg_simple_kernel, neg_mod_avx512, neg_mod, 1, b[l], 0)
#undef BARRET_ARGS_VEC_0
#undef BARRET_ARGS_VEC_1
#undef BARRET_ARGS_SCALAR_0
#undef BARRET_ARGS_SCALAR_1
#else
#define GENERATE_SIMPLE_KERNEL(NAME, OP, B_ACCESS, HAS_BARRET)                  \
  static void NAME(                                                             \
      const size_t L,                                                           \
      const size_t N,                                                           \
      uint64_t* c,                                                              \
      const uint64_t* a,                                                        \
      const uint64_t* b,                                                        \
      const uint64_t* mod BARRET_PARAMS_##HAS_BARRET) {                         \
    const int max_threads = omp_get_max_threads();                              \
    _Pragma("omp parallel for collapse(2) schedule(static) num_threads(max_threads)") \
    for (size_t l = 0; l < L; ++l) {                                            \
      for (size_t i = 0; i < N; ++i) {                                          \
        c[l * N + i] = OP(a[l * N + i], B_ACCESS, mod[l] BARRET_ARGS_##HAS_BARRET); \
      }                                                                         \
    }                                                                           \
  }
GENERATE_SIMPLE_KERNEL(vadd_simple_kernel, add_mod, b[l * N + i], 0)
GENERATE_SIMPLE_KERNEL(vsub_simple_kernel, sub_mod, b[l * N + i], 0)
GENERATE_SIMPLE_KERNEL(vmul_simple_kernel, mul_mod, b[l * N + i], 1)
GENERATE_SIMPLE_KERNEL(vadd_scalar_simple_kernel, add_mod, b[l], 0)
GENERATE_SIMPLE_KERNEL(vsub_scalar_simple_kernel, sub_mod, b[l], 0)
GENERATE_SIMPLE_KERNEL(vmul_scalar_simple_kernel, mul_mod, b[l], 1)
GENERATE_SIMPLE_KERNEL(vneg_simple_kernel, neg_mod, 0, 0)
#endif

#undef BARRET_PARAMS_0
#undef BARRET_PARAMS_1
#undef BARRET_ARGS_0
#undef BARRET_ARGS_1
#undef GENERATE_SIMPLE_KERNEL

/* kernels for 4D [num_cv, batch, L, N] tensors */

#define BARRET_PARAMS_0
#define BARRET_PARAMS_1 , const uint64_t* barret_mu

#define BARRET_ARGS_0
#define BARRET_ARGS_1 , barret_mu[l * 2], barret_mu[l * 2 + 1]

#ifdef USE_AVX512
#define BARRET_ARGS_VEC_0
#define BARRET_ARGS_VEC_1 \
  , _mm512_set1_epi64(barret_mu[l * 2]), _mm512_set1_epi64(barret_mu[l * 2 + 1])
#define BARRET_ARGS_SCALAR_0
#define BARRET_ARGS_SCALAR_1 , barret_mu[l * 2], barret_mu[l * 2 + 1]
#define GENERATE_KERNEL(NAME, OP_VEC, OP_SCALAR, B_IS_SCALAR, B_ACCESS, HAS_BARRET) \
  template <size_t NUM_CV>                                                       \
  static void NAME(                                                              \
      const size_t N,                                                            \
      const size_t cur_limbs,                                                    \
      const size_t LN_C,                                                         \
      const size_t LN_A,                                                         \
      const size_t LN_B,                                                         \
      const size_t BLN_C,                                                        \
      const size_t BLN_A,                                                        \
      const size_t BLN_B,                                                        \
      uint64_t* c,                                                               \
      const uint64_t* a,                                                         \
      const uint64_t* b,                                                         \
      const uint64_t* mod BARRET_PARAMS_##HAS_BARRET) {                          \
    const size_t batch = BLN_C / LN_C;                                           \
    const int max_threads = omp_get_max_threads();                               \
    _Pragma("omp parallel for collapse(2) schedule(static) num_threads(max_threads)") \
    for (size_t batch_id = 0; batch_id < batch; ++batch_id) {                    \
      for (size_t l = 0; l < cur_limbs; ++l) {                                   \
        size_t tid = 0;                                                          \
        const __m512i vec_mod = _mm512_set1_epi64(mod[l]);                      \
        for (; tid + 8 <= N; tid += 8) {                                         \
          for (size_t i = 0; i < NUM_CV; ++i) {                                  \
            const __m512i vec_a = _mm512_loadu_si512(                            \
                &a[i * BLN_A + batch_id * LN_A + l * N + tid]);                  \
            const __m512i vec_b = (B_IS_SCALAR) ? _mm512_set1_epi64(B_ACCESS)    \
                                                : _mm512_loadu_si512(&B_ACCESS);  \
            const __m512i vec_r =                                                 \
                OP_VEC(vec_a, vec_b, vec_mod BARRET_ARGS_VEC_##HAS_BARRET);      \
            _mm512_storeu_si512(&c[i * BLN_C + batch_id * LN_C + l * N + tid], vec_r); \
          }                                                                       \
        }                                                                         \
        for (; tid < N; ++tid) {                                                 \
          for (size_t i = 0; i < NUM_CV; ++i) {                                  \
            c[i * BLN_C + batch_id * LN_C + l * N + tid] =                       \
                OP_SCALAR(a[i * BLN_A + batch_id * LN_A + l * N + tid],          \
                          B_ACCESS,                                               \
                          mod[l] BARRET_ARGS_SCALAR_##HAS_BARRET);               \
          }                                                                       \
        }                                                                         \
      }                                                                           \
    }                                                                             \
  }
GENERATE_KERNEL(vadd_kernel, add_mod_avx512, add_mod, 0, b[i * BLN_B + batch_id * LN_B + l * N + tid], 0)
GENERATE_KERNEL(vsub_kernel, sub_mod_avx512, sub_mod, 0, b[i * BLN_B + batch_id * LN_B + l * N + tid], 0)
GENERATE_KERNEL(vmul_kernel, mul_mod_avx512, mul_mod, 0, b[i * BLN_B + batch_id * LN_B + l * N + tid], 1)
GENERATE_KERNEL(vadd_scalar_kernel, add_mod_avx512, add_mod, 1, b[l], 0)
GENERATE_KERNEL(vsub_scalar_kernel, sub_mod_avx512, sub_mod, 1, b[l], 0)
GENERATE_KERNEL(vmul_scalar_kernel, mul_mod_avx512, mul_mod, 1, b[l], 1)
GENERATE_KERNEL(vneg_kernel, neg_mod_avx512, neg_mod, 1, b[l], 0)
GENERATE_KERNEL(vadd_pt_broadcast_kernel, add_mod_avx512, add_mod, 0, b[l * N + tid], 0)
GENERATE_KERNEL(vadd_pt_pairwise_kernel, add_mod_avx512, add_mod, 0, b[batch_id * LN_B + l * N + tid], 0)
GENERATE_KERNEL(vmul_pt_broadcast_kernel, mul_mod_avx512, mul_mod, 0, b[l * N + tid], 1)
GENERATE_KERNEL(vmul_pt_pairwise_kernel, mul_mod_avx512, mul_mod, 0, b[batch_id * LN_B + l * N + tid], 1)
#undef BARRET_ARGS_VEC_0
#undef BARRET_ARGS_VEC_1
#undef BARRET_ARGS_SCALAR_0
#undef BARRET_ARGS_SCALAR_1
#else
#define GENERATE_KERNEL(NAME, OP, B_ACCESS, HAS_BARRET)                          \
  template <size_t NUM_CV>                                                       \
  static void NAME(                                                              \
      const size_t N,                                                            \
      const size_t cur_limbs,                                                    \
      const size_t LN_C,                                                         \
      const size_t LN_A,                                                         \
      const size_t LN_B,                                                         \
      const size_t BLN_C,                                                        \
      const size_t BLN_A,                                                        \
      const size_t BLN_B,                                                        \
      uint64_t* c,                                                               \
      const uint64_t* a,                                                         \
      const uint64_t* b,                                                         \
      const uint64_t* mod BARRET_PARAMS_##HAS_BARRET) {                          \
    const size_t batch = BLN_C / LN_C;                                           \
    const int max_threads = omp_get_max_threads();                               \
    _Pragma("omp parallel for collapse(3) schedule(static) num_threads(max_threads)") \
    for (size_t batch_id = 0; batch_id < batch; ++batch_id) {                    \
      for (size_t l = 0; l < cur_limbs; ++l) {                                   \
        for (size_t tid = 0; tid < N; ++tid) {                                   \
          for (size_t i = 0; i < NUM_CV; ++i) {                                  \
            c[i * BLN_C + batch_id * LN_C + l * N + tid] =                       \
                OP(a[i * BLN_A + batch_id * LN_A + l * N + tid],                 \
                   B_ACCESS,                                                    \
                   mod[l] BARRET_ARGS_##HAS_BARRET);                            \
          }                                                                      \
        }                                                                        \
      }                                                                          \
    }                                                                            \
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
#endif

#undef BARRET_PARAMS_0
#undef BARRET_PARAMS_1
#undef BARRET_ARGS_0
#undef BARRET_ARGS_1
#undef GENERATE_KERNEL

} // namespace fhe

namespace at::native {

/* simple kernel launchers */

#define BARRET_PARAMS_0
#define BARRET_PARAMS_1 , const uint64_t* barret_mu

#define BARRET_ARGS_0
#define BARRET_ARGS_1 , barret_mu

#define GENERATE_SIMPLE_FUNCTION(NAME, HAS_BARRET)                             \
  void NAME##_mod(                                                             \
      const size_t N,                                                          \
      int64_t l,                                                               \
      uint64_t* c,                                                             \
      const uint64_t* a,                                                       \
      const uint64_t* b,                                                       \
      const uint64_t* mod BARRET_PARAMS_##HAS_BARRET) {                        \
    fhe::NAME##_simple_kernel(l, N, c, a, b, mod BARRET_ARGS_##HAS_BARRET);     \
  }

GENERATE_SIMPLE_FUNCTION(vadd, 0)
GENERATE_SIMPLE_FUNCTION(vsub, 0)
GENERATE_SIMPLE_FUNCTION(vmul, 1)
GENERATE_SIMPLE_FUNCTION(vadd_scalar, 0)
GENERATE_SIMPLE_FUNCTION(vsub_scalar, 0)
GENERATE_SIMPLE_FUNCTION(vmul_scalar, 1)
GENERATE_SIMPLE_FUNCTION(vneg, 0)

#undef BARRET_PARAMS_0
#undef BARRET_PARAMS_1
#undef BARRET_ARGS_0
#undef BARRET_ARGS_1
#undef GENERATE_SIMPLE_FUNCTION

/* kernel launchers for 4D tensors */

#define BARRET_PARAMS_0
#define BARRET_PARAMS_1 , const uint64_t* barret_mu

#define BARRET_ARGS_0
#define BARRET_ARGS_1 , barret_mu

#define GENERATE_FUNCTION(NAME, HAS_BARRET)                               \
  void NAME##_mod(                                                        \
      const size_t num_cv,                                                \
      const size_t batch,                                                 \
      const size_t L_C,                                                   \
      const size_t L_A,                                                   \
      const size_t L_B,                                                   \
      const size_t B_NUMEL,                                               \
      const size_t N,                                                     \
      int64_t cur_limbs,                                                  \
      uint64_t* c,                                                        \
      const uint64_t* a,                                                  \
      const uint64_t* b,                                                  \
      const uint64_t* mod BARRET_PARAMS_##HAS_BARRET) {                   \
    TORCH_INTERNAL_ASSERT(cur_limbs >= 0);                                \
    TORCH_INTERNAL_ASSERT(static_cast<size_t>(cur_limbs) <= L_C);         \
    TORCH_INTERNAL_ASSERT(static_cast<size_t>(cur_limbs) <= L_A);         \
    TORCH_INTERNAL_ASSERT(static_cast<size_t>(cur_limbs) <= B_NUMEL);     \
    const auto LN_C = L_C * N;                                            \
    const auto LN_A = L_A * N;                                            \
    const auto LN_B = L_B * N;                                            \
    const auto BLN_C = batch * LN_C;                                      \
    const auto BLN_A = batch * LN_A;                                      \
    const auto BLN_B = batch * LN_B;                                      \
    if (num_cv == 1) {                                                    \
      fhe::NAME##_kernel<1>(                                              \
          N, static_cast<size_t>(cur_limbs), LN_C, LN_A, LN_B, BLN_C, BLN_A, BLN_B, c, a, b, mod BARRET_ARGS_##HAS_BARRET); \
    } else if (num_cv == 2) {                                             \
      fhe::NAME##_kernel<2>(                                              \
          N, static_cast<size_t>(cur_limbs), LN_C, LN_A, LN_B, BLN_C, BLN_A, BLN_B, c, a, b, mod BARRET_ARGS_##HAS_BARRET); \
    } else {                                                              \
      TORCH_INTERNAL_ASSERT(false, "Unsupported number of cvs");         \
    }                                                                     \
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
    const auto L_B = b.dim() >= 3 ? b.sizes()[2] : b.numel();                 \
    NAME##_mod(                                                                \
        num_cv,                                                                \
        batch,                                                                 \
        c.sizes()[2],                                                          \
        a.sizes()[2],                                                          \
        L_B,                                                                   \
        b.numel(),                                                             \
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
  Tensor NAME##_mod_cpu(                                                     \
      const Tensor& a,                                                       \
      const Tensor& b,                                                       \
      const Tensor& mod BARRET_PARAMS_##HAS_BARRET,                          \
      int64_t cur_limbs) {                                                   \
    Tensor c = at::empty(                                                    \
        {a.sizes()[0], a.sizes()[1], a.sizes()[2], a.sizes()[3]}, a.options()); \
    v##NAME##_template(c, a, b, mod BARRET_ARGS_##HAS_BARRET, cur_limbs);    \
    return c;                                                                \
  }                                                                          \
                                                                             \
  Tensor& NAME##_mod_cpu_(                                                   \
      Tensor& self,                                                          \
      const Tensor& other,                                                   \
      const Tensor& mod BARRET_PARAMS_##HAS_BARRET,                          \
      int64_t cur_limbs) {                                                   \
    v##NAME##_template(                                                      \
        self, self, other, mod BARRET_ARGS_##HAS_BARRET, cur_limbs);         \
    return self;                                                             \
  }

GENERATE_INTERFACE(add, 0)
GENERATE_INTERFACE(sub, 0)
GENERATE_INTERFACE(mul, 1)
GENERATE_INTERFACE(add_scalar, 0)
GENERATE_INTERFACE(sub_scalar, 0)
GENERATE_INTERFACE(mul_scalar, 1)
GENERATE_INTERFACE(neg, 0)

#define GENERATE_PT_INTERFACE(NAME, HAS_BARRET)                              \
  Tensor NAME##_cpu(                                                         \
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
