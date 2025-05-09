#include <ATen/Dispatch_v2.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/interned_strings.h>
#include <ATen/ops/copy.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>

#include "ATen/native/fhe/cpu/CommonOperation.h"
#include "ATen/native/fhe/cpu/Utils.h"

#pragma clang diagnostic ignored "-Wmissing-prototypes"

#define WORK_PER_THREAD (1)
#define WARP_SIZE (32)
#define NUM_WARPS (8)
#define BLOCK_SIZE (WARP_SIZE * NUM_WARPS)
#define WORK_PER_BLOCK (WORK_PER_THREAD * BLOCK_SIZE)

#define num_blocks(n) ((n + WORK_PER_BLOCK - 1) / WORK_PER_BLOCK)

namespace fhe {
inline void mulByMonomial_step1(
    uint64_t* out,
    const uint64_t* in,
    const uint64_t* qVec,
    int64_t l,
    int64_t N) {
  // Equivalent to the first kernel over a 2D grid (y = blockIdx.y, x = tid_x)
  for (int64_t row = 0; row < l; ++row) {
    const uint64_t q = qVec[row];
    const uint64_t* in_row = in + row * N;
    uint64_t* out_row = out + row * N;
    for (int64_t x = 0; x < N; ++x) {
      out_row[x] = q - in_row[x];
    }
  }
}

inline void mulByMonomial_step2(
    uint64_t* out,
    const uint64_t* in,
    const uint64_t* qVec,
    int64_t l,
    int64_t N,
    int64_t shift) {
  // Equivalent to the second kernel
  shift = shift % N;
  for (int64_t row = 0; row < l; ++row) {
    const uint64_t q = qVec[row];
    const uint64_t* in_row = in + row * N;
    uint64_t* out_row = out + row * N;
    for (int64_t x = 0; x < N; ++x) {
      if (x < shift) {
        out_row[x] = q - in_row[x + (N - shift)];
      } else {
        out_row[x] = in_row[x - shift];
      }
    }
  }
}

inline void mulByMonomial_step1_step2(
    uint64_t* out,
    const uint64_t* in,
    const uint64_t* qVec,
    int64_t l,
    int64_t N,
    int64_t shift) {
  // Equivalent to the combined kernel
  shift = shift % N;
  for (int64_t row = 0; row < l; ++row) {
    const uint64_t q = qVec[row];
    const uint64_t* in_row = in + row * N;
    uint64_t* out_row = out + row * N;
    for (int64_t x = 0; x < N; ++x) {
      if (x < shift) {
        out_row[x] = in_row[x + (N - shift)];
      } else {
        out_row[x] = q - in_row[x - shift];
      }
    }
  }
}
} // namespace fhe

namespace at::native {

static void mul_by_monomial_impl_cpu(
    uint64_t* out_ptr,
    const uint64_t* in_ptr,
    const uint64_t* primes_ptr,
    int64_t l,
    int64_t N,
    int64_t M,
    int64_t monomialDeg) {
  int64_t shift = monomialDeg % M;

  if (shift < N) {
    fhe::mulByMonomial_step2(
        out_ptr,
        in_ptr,
        primes_ptr,
        /*l=*/l,
        /*N=*/N,
        /*shift=*/shift);
  } else {
    shift = shift % N;
    fhe::mulByMonomial_step1_step2(
        out_ptr,
        in_ptr,
        primes_ptr,
        /*l=*/l,
        /*N=*/N,
        /*shift=*/shift);
  }
}

static void mul_by_monomial_template_cpu(
    Tensor& res,
    int64_t l,
    int64_t N,
    int64_t M,
    int64_t monomialDeg,
    int64_t level,
    const Tensor& param_primes,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_power_of_roots) {
  // 1) inverse NTT (in-place on res)
  uint64_t* res_ptr = reinterpret_cast<uint64_t*>(res.data_ptr<uint64_t>());
  iNTT_impl(
      res_ptr,
      res_ptr,
      0,
      l,
      l,
      level,
      N,
      inverse_power_of_roots_div_two,
      param_primes,
      inverse_scaled_power_of_roots_div_two);

  Tensor tmp = at::empty_like(res);
  uint64_t* tmp_ptr = reinterpret_cast<uint64_t*>(tmp.data_ptr<uint64_t>());
  const uint64_t* primes_ptr =
      reinterpret_cast<const uint64_t*>(param_primes.data_ptr<uint64_t>());

  // perform the CPU impl
  mul_by_monomial_impl_cpu(tmp_ptr, res_ptr, primes_ptr, l, N, M, monomialDeg);

  // copy back into res
  std::memcpy(res_ptr, tmp_ptr, size_t(l) * size_t(N) * sizeof(uint64_t));

  // 3) forward NTT (in-place on res)
  NTT_impl(
      res_ptr,
      res_ptr,
      0,
      l,
      N,
      param_power_of_roots_shoup,
      param_primes,
      param_power_of_roots);
}

Tensor mul_by_monomial_cpu(
    const Tensor& res,
    int64_t l,
    int64_t N,
    int64_t M,
    int64_t monomialDeg,
    int64_t level,
    const Tensor& param_primes,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_power_of_roots) {
  Tensor out = res.clone();
  mul_by_monomial_template_cpu(
      out,
      l,
      N,
      M,
      monomialDeg,
      level,
      param_primes,
      inverse_power_of_roots_div_two,
      inverse_scaled_power_of_roots_div_two,
      param_power_of_roots_shoup,
      param_power_of_roots);
  return out;
}

Tensor& mul_by_monomial_cpu_(
    Tensor& res,

    int64_t l,
    int64_t N,
    int64_t M,
    int64_t monomialDeg,
    int64_t level,
    const Tensor& param_primes,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_power_of_roots) {
  mul_by_monomial_template_cpu(
      res,
      l,
      N,
      M,
      monomialDeg,
      level,
      param_primes,
      inverse_power_of_roots_div_two,
      inverse_scaled_power_of_roots_div_two,
      param_power_of_roots_shoup,
      param_power_of_roots);
  return res;
}

Tensor& mul_by_monomial_cpu_out(
    const Tensor& res,
    int64_t l,
    int64_t N,
    int64_t M,
    int64_t monomialDeg,
    int64_t level,
    const Tensor& param_primes,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_power_of_roots,
    Tensor& out) {
  mul_by_monomial_template_cpu(
      out,
      l,
      N,
      M,
      monomialDeg,
      level,
      param_primes,
      inverse_power_of_roots_div_two,
      inverse_scaled_power_of_roots_div_two,
      param_power_of_roots_shoup,
      param_power_of_roots);
  return out;
}

} // namespace at::native
