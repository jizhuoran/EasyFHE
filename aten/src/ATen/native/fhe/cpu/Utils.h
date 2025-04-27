#pragma once

#include <cstdint>

#define STRIDED_LOOP_START(N, i) for (int i = 0; i < N; i++) {
#define STRIDED_LOOP_END }
#define TO_PTR(x) (x.data())

static inline __attribute__((always_inline)) void __syncthreads() {
  // Dummy implementation
}

namespace fhe {

using uint128_t = __uint128_t;

static inline uint64_t barret_reduction_128_64(
    __uint128_t x,
    uint64_t p,
    uint64_t mu,
    unsigned k // must satisfy 64 < k < 128
) {
  // split x
  uint64_t lo = uint64_t(x);
  uint64_t hi = uint64_t(x >> 64);

  // two 64*64 -> 128 multiplies
  __uint128_t m_lo = __uint128_t(lo) * mu;
  __uint128_t m_hi = __uint128_t(hi) * mu;

  // carry‐propagate the upper half of m_lo into m_hi
  uint64_t mid = uint64_t(m_lo >> 64);
  __uint128_t sum = m_hi + mid;

  // shift down by (k−64) to get the approximate quotient
  uint64_t q = uint64_t(sum >> (k - 64));

  // subtract and correct
  __uint128_t prod = __uint128_t(q) * p;
  uint64_t r = uint64_t(x - prod);
  return (r >= p) ? r - p : r;
}

#if defined(__BMI2__) && (defined(__x86_64__) || defined(_M_X64))
#include <immintrin.h>
inline uint64_t umul64hi(uint64_t a, uint64_t b) {
  uint64_t hi;
  _mulx_u64(a, b, &hi);
  return hi;
}
#else
inline uint64_t umul64hi(uint64_t a, uint64_t b) {
  return uint64_t((unsigned __int128)a * b >> 64);
}
#endif

inline uint64_t barret_reduction_64_64(
    uint64_t x,
    uint64_t modulus,
    uint64_t mu,
    unsigned shift_bits) {
  // q = floor(x * mu / 2^shift_bits)
  uint64_t q = umul64hi(x, mu) >> (shift_bits - 64);
  uint64_t r = x - q * modulus;
  return (r < modulus) ? r : (r - modulus);
}

static inline uint64_t mul_and_reduce_shoup(
    uint64_t a,
    uint64_t b,
    uint64_t bInv,
    uint64_t p) {
  uint64_t q = umul64hi(a, bInv);
  return a * b - q * p;
}

} // namespace fhe