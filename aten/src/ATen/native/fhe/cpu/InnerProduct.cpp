#include <ATen/Dispatch_v2.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/interned_strings.h>
#include <ATen/ops/copy.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/stack.h>
#include <ATen/ops/zeros.h>
#include <omp.h>
#ifdef USE_AVX512
#include <immintrin.h>
#endif
#include "ATen/native/fhe/cpu/Utils.h"

#pragma clang diagnostic ignored "-Wmissing-prototypes"

namespace at::native {

static void innerproduct_template(
    Tensor& out,
    const Tensor& in,
    const Tensor& bx,
    const Tensor& ax,
    int64_t curr_limbs,
    int64_t alpha,
    int64_t special_mod_start,
    int64_t L,
    int64_t N,
    const Tensor& primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    const Tensor& workspace) {
  (void)workspace;
  const int64_t beta = (curr_limbs + alpha - 1) / alpha;
  const int64_t sizeQP = primes.numel();
  const int64_t sizeP = sizeQP - L;
  const int64_t length = curr_limbs + sizeP;
  const int64_t mult_length = special_mod_start + sizeP;
  const int64_t prime_gap = L - curr_limbs;
  const int64_t swk_gap = special_mod_start - curr_limbs;
  TORCH_INTERNAL_ASSERT(
      special_mod_start >= curr_limbs,
      "special_mod_start must be >= curr_limbs");
  TORCH_INTERNAL_ASSERT(
      special_mod_start <= L,
      "special_mod_start must be <= L");
  TORCH_INTERNAL_ASSERT(bx.dim() == 3, "bx must be [beta, mult_length, N]");
  TORCH_INTERNAL_ASSERT(ax.dim() == 3, "ax must be [beta, mult_length, N]");
  TORCH_INTERNAL_ASSERT(
      bx.sizes() == ax.sizes(),
      "bx and ax must have identical shapes");
  TORCH_INTERNAL_ASSERT(
      bx.size(0) >= beta,
      "bx/ax beta dimension must be >= ceil(curr_limbs / alpha)");
  TORCH_INTERNAL_ASSERT(
      bx.size(1) >= mult_length,
      "bx/ax modulus dimension must be >= special_mod_start + sizeP");
  TORCH_INTERNAL_ASSERT(bx.size(2) == N, "bx/ax last dimension must equal N");

  const auto* in_ptr = in.data_ptr<uint64_t>();
  const auto* ax_ptr = ax.data_ptr<uint64_t>();
  const auto* bx_ptr = bx.data_ptr<uint64_t>();
  auto* out_bx_ptr = out[0].data_ptr<uint64_t>();
  auto* out_ax_ptr = out[1].data_ptr<uint64_t>();
  const auto* primes_ptr = primes.data_ptr<uint64_t>();
  const auto* barret_ratio_ptr = barret_ratio.data_ptr<uint64_t>();
  const auto* barret_k_ptr = barret_k.data_ptr<uint64_t>();

  const auto batch = out.sizes()[1];
  const int64_t batch_stride = beta * length * N;
  const int64_t out_stride = length * N;

  const int max_threads = omp_get_max_threads();
#pragma omp parallel for collapse(2) schedule(static) num_threads(max_threads)
  for (int64_t batch_id = 0; batch_id < batch; ++batch_id) {
    for (int64_t idx = 0; idx < length; ++idx) {
      const int64_t prime_idx = (idx < curr_limbs) ? 0 : prime_gap;
      const int64_t swk_idx = (idx < curr_limbs) ? 0 : swk_gap;
      const int64_t reduce_prime_idx = idx + prime_idx;
      const int64_t swk_prime_idx = idx + swk_idx;
      const auto prime = primes_ptr[reduce_prime_idx];
      const auto ratio = barret_ratio_ptr[reduce_prime_idx];
      const auto k = barret_k_ptr[reduce_prime_idx];

#ifdef USE_AVX512
      const __m512i prime_vec = _mm512_set1_epi64(prime);
      const __m512i ratio_vec = _mm512_set1_epi64(ratio);
      int64_t n = 0;
      for (; n + 8 <= N; n += 8) {
        __m512i accum_ax_lo = _mm512_setzero_si512();
        __m512i accum_ax_hi = _mm512_setzero_si512();
        __m512i accum_bx_lo = _mm512_setzero_si512();
        __m512i accum_bx_hi = _mm512_setzero_si512();

        for (int64_t beta_idx = 0; beta_idx < beta; ++beta_idx) {
          const int64_t in_off =
              batch_id * batch_stride + (beta_idx * length + idx) * N + n;
          const int64_t swk_off =
              (beta_idx * mult_length + swk_prime_idx) * N + n;

          const __m512i op1_vec = _mm512_loadu_si512(&in_ptr[in_off]);
          const __m512i op2_ax_vec = _mm512_loadu_si512(&ax_ptr[swk_off]);
          const __m512i op2_bx_vec = _mm512_loadu_si512(&bx_ptr[swk_off]);

          const __m512i mul_ax_lo = _mm512_mullo_epi64(op1_vec, op2_ax_vec);
          const __m512i mul_ax_hi = fhe::avx_umul64hi(op1_vec, op2_ax_vec);
          const __m512i mul_bx_lo = _mm512_mullo_epi64(op1_vec, op2_bx_vec);
          const __m512i mul_bx_hi = fhe::avx_umul64hi(op1_vec, op2_bx_vec);

          fhe::avx512_add_u128(
              accum_ax_lo, accum_ax_hi, mul_ax_lo, mul_ax_hi, accum_ax_lo, accum_ax_hi);
          fhe::avx512_add_u128(
              accum_bx_lo, accum_bx_hi, mul_bx_lo, mul_bx_hi, accum_bx_lo, accum_bx_hi);
        }

        const __m512i out_ax_vec = fhe::barret_reduction_128_64_avx512(
            accum_ax_lo, accum_ax_hi, prime_vec, ratio_vec, static_cast<unsigned>(k));
        const __m512i out_bx_vec = fhe::barret_reduction_128_64_avx512(
            accum_bx_lo, accum_bx_hi, prime_vec, ratio_vec, static_cast<unsigned>(k));

        _mm512_storeu_si512(&out_ax_ptr[batch_id * out_stride + idx * N + n], out_ax_vec);
        _mm512_storeu_si512(&out_bx_ptr[batch_id * out_stride + idx * N + n], out_bx_vec);
      }
      for (; n < N; ++n) {
        __uint128_t accum_ax{0};
        __uint128_t accum_bx{0};
        for (int64_t beta_idx = 0; beta_idx < beta; ++beta_idx) {
          const int64_t in_off =
              batch_id * batch_stride + (beta_idx * length + idx) * N + n;
          const int64_t swk_off =
              (beta_idx * mult_length + swk_prime_idx) * N + n;

          accum_ax += static_cast<__uint128_t>(in_ptr[in_off]) * ax_ptr[swk_off];
          accum_bx += static_cast<__uint128_t>(in_ptr[in_off]) * bx_ptr[swk_off];
        }

        out_ax_ptr[batch_id * out_stride + idx * N + n] = fhe::barret_reduction_128_64(
            accum_ax, prime, ratio, static_cast<unsigned>(k));
        out_bx_ptr[batch_id * out_stride + idx * N + n] = fhe::barret_reduction_128_64(
            accum_bx, prime, ratio, static_cast<unsigned>(k));
      }
#else
      for (int64_t n = 0; n < N; ++n) {
        __uint128_t accum_ax{0};
        __uint128_t accum_bx{0};
        for (int64_t beta_idx = 0; beta_idx < beta; ++beta_idx) {
          const int64_t in_off =
              batch_id * batch_stride + (beta_idx * length + idx) * N + n;
          const int64_t swk_off =
              (beta_idx * mult_length + swk_prime_idx) * N + n;

          const uint64_t op1 = in_ptr[in_off];
          const uint64_t op2_ax = ax_ptr[swk_off];
          const uint64_t op2_bx = bx_ptr[swk_off];
          accum_ax += static_cast<__uint128_t>(op1) * op2_ax;
          accum_bx += static_cast<__uint128_t>(op1) * op2_bx;
        }

        out_ax_ptr[batch_id * out_stride + idx * N + n] = fhe::barret_reduction_128_64(
            accum_ax, prime, ratio, static_cast<unsigned>(k));
        out_bx_ptr[batch_id * out_stride + idx * N + n] = fhe::barret_reduction_128_64(
            accum_bx, prime, ratio, static_cast<unsigned>(k));
      }
#endif
    }
  }
}

Tensor innerproduct_cpu(
    const Tensor& in,
    const Tensor& bx,
    const Tensor& ax,
    int64_t curr_limbs,
    int64_t alpha,
    int64_t special_mod_start,
    int64_t L,
    int64_t N,
    const Tensor& primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    const Tensor& workspace) {
  TORCH_INTERNAL_ASSERT(in.dim() == 4);
  const auto num_cv = in.sizes()[0];
  TORCH_INTERNAL_ASSERT(num_cv == 1, "innerproduct_cpu expects num_cv == 1");
  const auto batch = in.sizes()[1];

  const int64_t sizeQP = primes.numel();
  const int64_t sizeP = sizeQP - L;
  auto out = at::empty({2, batch, curr_limbs + sizeP, N}, in.options());

  innerproduct_template(
      out,
      in,
      bx,
      ax,
      curr_limbs,
      alpha,
      special_mod_start,
      L,
      N,
      primes,
      barret_ratio,
      barret_k,
      workspace);

  return out;
}

static void innerproduct_broadcast_cipher_template(
    Tensor& out,
    const Tensor& in,
    const TensorList& swks,
    int64_t curr_limbs,
    int64_t alpha,
    int64_t L,
    int64_t N,
    const Tensor& special_mod_start,
    const Tensor& primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    const Tensor& workspace) {
  (void)workspace;
  const int64_t beta = (curr_limbs + alpha - 1) / alpha;
  const int64_t sizeQP = primes.numel();
  const int64_t sizeP = sizeQP - L;
  const int64_t length = curr_limbs + sizeP;

  const auto* in_ptr = in.data_ptr<uint64_t>();
  auto* out_ptr = out.data_ptr<uint64_t>();
  const auto* primes_ptr = primes.data_ptr<uint64_t>();
  const auto* barret_ratio_ptr = barret_ratio.data_ptr<uint64_t>();
  const auto* barret_k_ptr = barret_k.data_ptr<uint64_t>();
  const auto* special_mod_start_ptr = special_mod_start.data_ptr<int64_t>();

  TORCH_INTERNAL_ASSERT(swks.size() > 0, "swks must not be empty");
  TORCH_INTERNAL_ASSERT(
      special_mod_start.device() == in.device(),
      "special_mod_start must be on same device as in");
  TORCH_INTERNAL_ASSERT(
      special_mod_start.numel() == static_cast<int64_t>(swks.size()),
      "special_mod_start numel must equal swks.size()");
  for (int64_t swk_idx = 0; swk_idx < static_cast<int64_t>(swks.size()); ++swk_idx) {
    TORCH_INTERNAL_ASSERT(
        swks[swk_idx].dim() == 4,
        "swk tensor must have shape [2, beta, mult_length, N]");
    const auto sizes = swks[swk_idx].sizes();
    TORCH_INTERNAL_ASSERT(sizes[0] == 2, "swk first dimension must be 2 (bx/ax)");
    TORCH_INTERNAL_ASSERT(
        special_mod_start_ptr[swk_idx] >= curr_limbs,
        "special_mod_start[swk_idx] must be >= curr_limbs");
    TORCH_INTERNAL_ASSERT(
        special_mod_start_ptr[swk_idx] <= L,
        "special_mod_start[swk_idx] must be <= L");
    TORCH_INTERNAL_ASSERT(
        sizes[1] >= beta,
        "swk beta dimension must be >= ceil(curr_limbs / alpha)");
    TORCH_INTERNAL_ASSERT(sizes[3] == N, "swk last dimension must equal N");
    const int64_t expected_mult_length = special_mod_start_ptr[swk_idx] + sizeP;
    TORCH_INTERNAL_ASSERT(
        sizes[2] >= expected_mult_length,
        "swk modulus dimension must be >= special_mod_start[swk_idx] + sizeP");
  }

  const int max_threads = omp_get_max_threads();
#pragma omp parallel for collapse(3) schedule(static) num_threads(max_threads)
  for (int64_t swk_idx = 0; swk_idx < static_cast<int64_t>(swks.size()); ++swk_idx) {
    for (int64_t idx = 0; idx < length; ++idx) {
      for (int64_t n = 0; n < N; ++n) {
        const int64_t swk_special_mod_start = special_mod_start_ptr[swk_idx];
        const int64_t prime_gap = L - curr_limbs;
        const int64_t swk_gap = swk_special_mod_start - curr_limbs;
        const int64_t mult_length = swk_special_mod_start + sizeP;
        const int64_t prime_idx = (idx < curr_limbs) ? 0 : prime_gap;
        const int64_t swk_idx_delta = (idx < curr_limbs) ? 0 : swk_gap;
        const int64_t reduce_prime_idx = idx + prime_idx;
        const int64_t swk_prime_idx = idx + swk_idx_delta;

        __uint128_t accum_ax{0};
        __uint128_t accum_bx{0};

        const int64_t swk_off =
            swks[swk_idx].sizes()[1] * swks[swk_idx].sizes()[2] * swks[swk_idx].sizes()[3];
        const auto* swk_ptr = swks[swk_idx].data_ptr<uint64_t>();
        for (int64_t beta_idx = 0; beta_idx < beta; ++beta_idx) {
          const int64_t in_off = (beta_idx * length + idx) * N + n;
          const int64_t swk_off_base = (beta_idx * mult_length + swk_prime_idx) * N + n;

          const uint64_t op1 = in_ptr[in_off];
          const uint64_t op2_bx = swk_ptr[swk_off_base];
          const uint64_t op2_ax = swk_ptr[swk_off_base + swk_off];

          accum_ax += static_cast<__uint128_t>(op1) * op2_ax;
          accum_bx += static_cast<__uint128_t>(op1) * op2_bx;
        }

        const auto prime = primes_ptr[reduce_prime_idx];
        const auto ratio = barret_ratio_ptr[reduce_prime_idx];
        const auto k = barret_k_ptr[reduce_prime_idx];

        const uint64_t res_ax = fhe::barret_reduction_128_64(
            accum_ax, prime, ratio, static_cast<unsigned>(k));
        const uint64_t res_bx = fhe::barret_reduction_128_64(
            accum_bx, prime, ratio, static_cast<unsigned>(k));

        const int64_t out_base = swk_idx * 2 * length * N;
        out_ptr[out_base + 0 * length * N + idx * N + n] = res_bx;
        out_ptr[out_base + 1 * length * N + idx * N + n] = res_ax;
      }
    }
  }
}

Tensor innerproduct_broadcast_cipher_cpu(
    const Tensor& in,
    TensorList swks,
    int64_t curr_limbs,
    int64_t alpha,
    int64_t L,
    int64_t N,
    const Tensor& special_mod_start,
    const Tensor& primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    const Tensor& workspace) {
  TORCH_INTERNAL_ASSERT(in.dim() == 4);
  TORCH_INTERNAL_ASSERT(in.sizes()[0] == 1, "innerproduct_broadcast_cipher expects num_cv == 1");
  TORCH_INTERNAL_ASSERT(in.sizes()[1] == 1, "innerproduct_broadcast_cipher expects batch == 1");

  const int64_t sizeQP = primes.numel();
  const int64_t sizeP = sizeQP - L;
  auto out = at::empty({static_cast<int64_t>(swks.size()), 2, curr_limbs + sizeP, N}, in.options());

  innerproduct_broadcast_cipher_template(
      out,
      in,
      swks,
      curr_limbs,
      alpha,
      L,
      N,
      special_mod_start,
      primes,
      barret_ratio,
      barret_k,
      workspace);

  return out;
}

} // namespace at::native
