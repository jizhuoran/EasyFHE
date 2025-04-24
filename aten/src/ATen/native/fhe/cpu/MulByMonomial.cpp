#include <ATen/Dispatch_v2.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/interned_strings.h>
#include <ATen/ops/copy.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>

#include "ATen/native/fhe/cpu/KeySwitch.h"
#include "ATen/native/fhe/cpu/Utils.h"

#pragma clang diagnostic ignored "-Wmissing-prototypes"

#define WORK_PER_THREAD (1)
#define WARP_SIZE (32)
#define NUM_WARPS (8)
#define BLOCK_SIZE (WARP_SIZE * NUM_WARPS)
#define WORK_PER_BLOCK (WORK_PER_THREAD * BLOCK_SIZE)

#define num_blocks(n) ((n + WORK_PER_BLOCK - 1) / WORK_PER_BLOCK)

namespace fhe {
// __global__ void mulByMonomialKernel_step1(
//     uint64_t* res,
//     uint64_t* qVec,
//     uint64_t* tmp,
//     long l,
//     long N) {
//   STRIDED_LOOP_START(l * N, idx)
//   if (idx < l * N) {
//     long i = idx / N;
//     long n = idx % N;
//       tmp[idx] = qVec[i] - res[idx];
//   }
//   STRIDED_LOOP_END;
// }

// __global__ void mulByMonomialKernel_step2(
//     uint64_t* res,
//     uint64_t* qVec,
//     uint64_t* tmp,
//     long l,
//     long N,
//     long shift) {
//   STRIDED_LOOP_START(l * N, idx)
//   if (idx < l * N) {
//     long i = idx / N;
//     long n = idx % N;
//     shift %= N;
//     if (n < shift) {
//       res[idx] = qVec[i] - tmp[idx +(N - shift)];
//     } else {
//       res[idx] = tmp[idx - shift];
//     }
//   }
//   STRIDED_LOOP_END;
// }
} // namespace fhe

namespace at::native {

static void mul_by_monomial_impl(
    uint64_t* res_ptr,
    const Tensor& primes,
    Tensor& tmp,
    int64_t l,
    int64_t N,
    int64_t M,
    int64_t monomialDeg) {
  int64_t shift = monomialDeg % M;
  AT_DISPATCH_V2(
      tmp.scalar_type(),
      "mul_by_monomial_impl",
      AT_WRAP([&]() {
        auto primes_ptr = reinterpret_cast<uint64_t*>(primes.data_ptr<uint64_t>());
        auto tmp_ptr = reinterpret_cast<uint64_t*>(tmp.data_ptr<uint64_t>());
        const int block_dim = 256;
        const int grid_dim = N * l / block_dim;
        // if (shift > N || shift == N) {
        //   fhe::mulByMonomialKernel_step1<<<grid_dim, block_dim, 0, stream>>>(
        //     res_ptr, primes_ptr, tmp_ptr, l, N);
        // }
        // fhe::mulByMonomialKernel_step2<<<grid_dim, block_dim, 0, stream>>>(
        //    res_ptr, primes_ptr, tmp_ptr, l, N, shift);
      }),
      kUInt64);
}

static void mul_by_monomial_template(
    Tensor& res,
    const Tensor& param_primes,
    int64_t l,
    int64_t N,
    int64_t M,
    int64_t monomialDeg,
    int64_t level,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_power_of_roots) {
  auto res_ptr = reinterpret_cast<uint64_t*>(res.data_ptr<uint64_t>());
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

  Tensor temp = res.clone();
  mul_by_monomial_impl(res_ptr, param_primes, temp, l, N, M, monomialDeg);

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
    const Tensor& param_primes,
    int64_t l,
    int64_t N,
    int64_t M,
    int64_t monomialDeg,
    int64_t level,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_power_of_roots) {
  Tensor out = res.clone();
  mul_by_monomial_template(
      out,
      param_primes,
      l,
      N,
      M,
      monomialDeg,
      level,
      inverse_power_of_roots_div_two,
      inverse_scaled_power_of_roots_div_two,
      param_power_of_roots_shoup,
      param_power_of_roots);
  return out;
}

Tensor& mul_by_monomial_cpu_(
    Tensor& res,
    const Tensor& param_primes,
    int64_t l,
    int64_t N,
    int64_t M,
    int64_t monomialDeg,
    int64_t level,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_power_of_roots) {
  mul_by_monomial_template(
      res,
      param_primes,
      l,
      N,
      M,
      monomialDeg,
      level,
      inverse_power_of_roots_div_two,
      inverse_scaled_power_of_roots_div_two,
      param_power_of_roots_shoup,
      param_power_of_roots);
  return res;
}

Tensor& mul_by_monomial_cpu_out(
    const Tensor& res,
    const Tensor& param_primes,
    int64_t l,
    int64_t N,
    int64_t M,
    int64_t monomialDeg,
    int64_t level,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_power_of_roots,
    Tensor& out) {
  mul_by_monomial_template(
      out,
      param_primes,
      l,
      N,
      M,
      monomialDeg,
      level,
      inverse_power_of_roots_div_two,
      inverse_scaled_power_of_roots_div_two,
      param_power_of_roots_shoup,
      param_power_of_roots);
  return out;
}

} // namespace at::native
