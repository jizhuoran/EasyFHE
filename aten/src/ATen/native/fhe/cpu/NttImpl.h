#pragma once
#include <ATen/Tensor.h>
#include <cstdint>
#include "ATen/native/fhe/cpu/Utils.h"
namespace fhe {

static inline __attribute__((always_inline)) void butt_intt_local(
    uint64_t& x,
    uint64_t& y,
    const uint64_t& w,
    const uint64_t& w_,
    const uint64_t& p) {
  const uint64_t two_p = 2 * p;
  const uint64_t T = two_p - y + x;
  uint64_t new_x = x + y;
  if (new_x >= two_p)
    new_x -= two_p;
  if (T & 1)
    new_x += p;
  x = (new_x >> 1);
  y = mul_and_reduce_shoup(T, w, w_, p);
}

static inline __attribute__((always_inline)) void butt_ntt_local(
    uint64_t& a,
    uint64_t& b,
    const uint64_t& w,
    const uint64_t& w_,
    const uint64_t p) {
  uint64_t two_p = 2 * p;
  uint64_t U = mul_and_reduce_shoup(b, w, w_, p);
  if (a >= two_p)
    a -= two_p;
  b = a + (two_p - U);
  a += U;
}
static int GetMSB(int64_t x) {
  if (x == 0)
    return -1; // No set bit, return -1

  int position = 0;
  while (x > 0) {
    x >>= 1; // Shift right by 1 bit
    position++; // Increment the position
  }

  return position; // The MSB is 1 less than the number of shifts
}
} // namespace fhe

namespace at::native {

void iNTT_impl(
    uint64_t* in_ptr,
    uint64_t* out_ptr,
    int64_t start_prime_idx,
    int64_t batch,
    int64_t curr_limbs,
    int64_t level,
    int64_t param_degree,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& param_primes,
    const Tensor& inverse_scaled_power_of_roots_div_two);

void NTT_impl(
    uint64_t* inout_ptr,
    int64_t start_prime_idx,
    int64_t batch,
    int64_t param_degree,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_primes,
    const Tensor& param_power_of_roots);

void NTT_except_some_range_impl(
    uint64_t* op_ptr,
    int64_t start_prime_idx,
    int64_t batch,
    int64_t param_degree,
    int64_t excluded_range_start,
    int64_t excluded_range_size,
    int64_t curr_limbs,
    int64_t level,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_primes,
    const Tensor& param_power_of_roots);

} // namespace at::native
