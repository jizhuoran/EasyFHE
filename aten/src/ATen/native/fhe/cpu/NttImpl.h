#pragma once
#include <omp.h>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <tuple>
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
    const Tensor& inverse_scaled_power_of_roots_div_two) {
  const int max_threads = omp_get_max_threads();
  omp_set_num_threads(max_threads);
  const uint64_t n = param_degree;
  auto inverse_power_of_roots_div_two_ptr = reinterpret_cast<uint64_t*>(
      inverse_power_of_roots_div_two.data_ptr<uint64_t>());
  auto param_primes_ptr =
      reinterpret_cast<uint64_t*>(param_primes.data_ptr<uint64_t>());
  auto inverse_scaled_power_of_roots_div_two_ptr = reinterpret_cast<uint64_t*>(
      inverse_scaled_power_of_roots_div_two.data_ptr<uint64_t>());
  int gap = level - curr_limbs;
#pragma omp parallel for schedule(static) num_threads(max_threads)
  for (int bach = 0; bach < batch; ++bach) {
    uint64_t primeidx = start_prime_idx + bach;
    uint64_t prime_idx =
        primeidx + ((primeidx >= 0 && primeidx < curr_limbs) ? 0 : gap);
    uint64_t modulus = param_primes_ptr[prime_idx];
    uint64_t base_prime_idx = prime_idx * param_degree;
    uint64_t base = primeidx * param_degree;

    for (uint32_t m = n >> 1, t = 1, logt = 1; m > 1;
         m >>= 1, t <<= 1, ++logt) {
      for (uint32_t i = 0; i < m; ++i) {
        auto omega = inverse_power_of_roots_div_two_ptr[i + m + base_prime_idx];
        auto preconOmega =
            inverse_scaled_power_of_roots_div_two_ptr[i + m + base_prime_idx];
        for (uint32_t j1 = i << logt, j2 = j1 + t; j1 < j2; ++j1) {
          auto loVal = out_ptr[j1 + 0 + base];
          auto hiVal = out_ptr[j1 + t + base];
          fhe::butt_intt_local(loVal, hiVal, omega, preconOmega, modulus);
          out_ptr[j1 + 0 + base] = loVal;
          out_ptr[j1 + t + base] = hiVal;
        }
      }
    }

    auto omega = inverse_power_of_roots_div_two_ptr[1 + base_prime_idx];
    auto preconOmega =
        inverse_scaled_power_of_roots_div_two_ptr[1 + base_prime_idx];
    uint32_t j2 = n >> 1;
    for (uint32_t j1 = 0; j1 < j2; ++j1) {
      auto loVal = (out_ptr)[j1 + base];
      auto hiVal = (out_ptr)[j1 + j2 + base];
      fhe::butt_intt_local(loVal, hiVal, omega, preconOmega, modulus);
      for (int i = 0; i < 8; i++) {
        if (loVal > modulus) {
          loVal -= modulus;
        }
        if (hiVal > modulus) {
          hiVal -= modulus;
        }
      }
      (out_ptr)[j1 + base] = loVal;
      (out_ptr)[j1 + j2 + base] = hiVal;
    }
  }
}

int GetMSB(int64_t x) {
  if (x == 0)
    return -1; // No set bit, return -1

  int position = 0;
  while (x > 0) {
    x >>= 1; // Shift right by 1 bit
    position++; // Increment the position
  }

  return position; // The MSB is 1 less than the number of shifts
}

void NTT_impl(
    uint64_t* in_ptr,
    uint64_t* out_ptr,
    int64_t start_prime_idx,
    int64_t batch,
    int64_t param_degree,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_primes,
    const Tensor& param_power_of_roots) {
  const int max_threads = omp_get_max_threads();
  omp_set_num_threads(max_threads);
  auto param_power_of_roots_shoup_ptr = reinterpret_cast<uint64_t*>(
      param_power_of_roots_shoup
          .data_ptr<uint64_t>()); // preconrootOfUnityTable
  auto param_primes_ptr =
      reinterpret_cast<uint64_t*>(param_primes.data_ptr<uint64_t>()); // modulo
  auto param_power_of_roots_ptr = reinterpret_cast<uint64_t*>(
      param_power_of_roots.data_ptr<uint64_t>()); // rootOfUnityTable
  const int64_t n = param_degree >> 1;
#pragma omp parallel for schedule(static) num_threads(max_threads)
  for (int bach = 0; bach < batch; ++bach) {
    auto modulus = param_primes_ptr[start_prime_idx + bach];
    auto primeidx = (start_prime_idx + bach);
    auto base = primeidx * param_degree;
    for (uint32_t m = 1, t = n, logt = GetMSB(t); m < n;
         m <<= 1, t >>= 1, --logt) {
      for (uint32_t i = 0; i < m; ++i) {
        auto omega = param_power_of_roots_ptr[i + m + base]; // S
        auto preconOmega =
            param_power_of_roots_shoup_ptr[i + m + base]; // NEEDED IN COMPUTE
                                                          // F[j+t]*S MOD Q
        for (uint32_t j1 = (i << logt), j2 = j1 + t; j1 < j2; ++j1) {
          uint64_t a1 = (out_ptr)[j1 + 0 + base];
          uint64_t b1 = (out_ptr)[j1 + t + base];
          fhe::butt_ntt_local(a1, b1, omega, preconOmega, modulus);
          (out_ptr)[j1 + 0 + base] = a1;
          (out_ptr)[j1 + t + base] = b1;
        }
      }
    }
#pragma omp parallel for schedule(static) num_threads(max_threads)
    for (uint32_t i = 0; i < (n << 1); i += 2) {
      auto omega = param_power_of_roots_ptr[(i >> 1) + n + base];
      auto preconOmega = param_power_of_roots_shoup_ptr[(i >> 1) + n + base];
      uint64_t a1 = (out_ptr)[i + 0 + base];
      uint64_t b1 = (out_ptr)[i + 1 + base];
      fhe::butt_ntt_local(a1, b1, omega, preconOmega, modulus);
      for (int a = 0; a < 3; a++) {
        if (b1 > modulus) {
          b1 -= modulus;
        }
        if (a1 > modulus) {
          a1 -= modulus;
        }
      }
      (out_ptr)[i + 0 + base] = a1;
      (out_ptr)[i + 1 + base] = b1;
    }
  }
}

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
    const Tensor& param_power_of_roots) {
  auto excluded_range_end = excluded_range_start + excluded_range_size;
  const int max_threads = omp_get_max_threads();
  omp_set_num_threads(max_threads);

  auto param_power_of_roots_shoup_ptr = reinterpret_cast<uint64_t*>(
      param_power_of_roots_shoup.data_ptr<uint64_t>());
  auto param_primes_ptr =
      reinterpret_cast<uint64_t*>(param_primes.data_ptr<uint64_t>());
  auto param_power_of_roots_ptr =
      reinterpret_cast<uint64_t*>(param_power_of_roots.data_ptr<uint64_t>());
  int gap = level - curr_limbs;
  const int64_t n = param_degree >> 1;
#pragma omp parallel for schedule(static) num_threads(max_threads)
  for (int bach = 0; bach < batch; ++bach) {
    uint64_t primeidx = batch - 1 - bach;
    +start_prime_idx;
    if (primeidx >= excluded_range_start && primeidx < excluded_range_end)
      continue;
    uint64_t prime_idx =
        primeidx + ((primeidx >= 0 && primeidx < curr_limbs) ? 0 : gap);
    uint64_t modulus = param_primes_ptr[prime_idx];
    uint64_t base_prime_idx = prime_idx * param_degree;
    uint64_t base = primeidx * param_degree;
    for (uint32_t m = 1, t = n, logt = GetMSB(t); m < n;
         m <<= 1, t >>= 1, --logt) {
      for (uint32_t i = 0; i < m; ++i) {
        auto omega = param_power_of_roots_ptr[i + m + base_prime_idx]; // S
        auto preconOmega = param_power_of_roots_shoup_ptr
            [i + m + base_prime_idx]; // NEEDED IN COMPUTE F[j+t]*S MOD Q
        for (uint32_t j1 = (i << logt), j2 = j1 + t; j1 < j2; ++j1) {
          uint64_t a1 = (op_ptr)[j1 + 0 + base];
          uint64_t b1 = (op_ptr)[j1 + t + base];
          fhe::butt_ntt_local(a1, b1, omega, preconOmega, modulus);
          (op_ptr)[j1 + 0 + base] = a1;
          (op_ptr)[j1 + t + base] = b1;
        }
      }
    }
    for (uint32_t i = 0; i < (n << 1); i += 2) {
      auto omega = param_power_of_roots_ptr[(i >> 1) + n + base_prime_idx];
      auto preconOmega =
          param_power_of_roots_shoup_ptr[(i >> 1) + n + base_prime_idx];
      uint64_t a1 = (op_ptr)[i + 0 + base];
      uint64_t b1 = (op_ptr)[i + 1 + base];
      fhe::butt_ntt_local(a1, b1, omega, preconOmega, modulus);
      for (int a = 0; a < 3; a++) {
        if (b1 > modulus) {
          b1 -= modulus;
        }
        if (a1 > modulus) {
          a1 -= modulus;
        }
      }
      (op_ptr)[i + 0 + base] = a1;
      (op_ptr)[i + 1 + base] = b1;
    }
  }
}



} // namespace at::native
