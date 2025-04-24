#pragma once

#include <ATen/cuda/CUDAContext.h>

namespace fhe {

__device__ __forceinline__ uint64_t
neg_mod(uint64_t x, uint64_t null, uint64_t mod) {
  return x == 0 ? 0 : mod - x;
}

__device__ __forceinline__ uint64_t
add_mod(uint64_t a, uint64_t b, uint64_t mod) {
  uint64_t res = a + b;
  return res >= mod ? res - mod : res;
}

__device__ __forceinline__ uint64_t
sub_mod(uint64_t a, uint64_t b, uint64_t mod) {
  return a >= b ? a - b : a + mod - b;
}

__device__ __forceinline__ uint64_t mul_mod(
    uint64_t a,
    uint64_t b,
    uint64_t mod,
    uint64_t barret_mu0,
    uint64_t barret_mu1) {
  uint64_t res;
  asm("{"
      " .reg .u64 tmp;"
      " .reg .u64 lo, hi;"
      // 128-bit multiply
      " mul.lo.u64 lo, %1, %2;"
      " mul.hi.u64 hi, %1, %2;"
      // Multiply input and const_ratio
      // Round 1
      " mul.hi.u64 tmp, lo, %3;"
      " mad.lo.cc.u64 tmp, lo, %4, tmp;"
      " madc.hi.u64 %0, lo, %4, 0;"
      // Round 2
      " mad.lo.cc.u64 tmp, hi, %3, tmp;"
      " madc.hi.u64 %0, hi, %3, %0;"
      // This is all we care about
      " mad.lo.u64 %0, hi, %4, %0;"
      // Barrett subtraction
      " mul.lo.u64 %0, %0, %5;"
      " sub.u64 %0, lo, %0;"
      "}"
      : "=l"(res)
      : "l"(a), "l"(b), "l"(barret_mu0), "l"(barret_mu1), "l"(mod));
  return res >= mod ? res - mod : res;
}

} // namespace fhe


namespace fhe {
struct uint128_t {
  uint64_t hi;
  uint64_t lo;
};

__inline__ __device__ uint128_t
mult_64_64_128(const uint64_t op1, const uint64_t op2) {
  uint128_t res;
  res.lo = op1 * op2;
  res.hi = __umul64hi(op1, op2);
  return res;
}

__inline__ __device__ void inplace_add_128_128(
    const uint128_t op1,
    uint128_t& res) {
  asm("add.cc.u64 %1, %3, %1;"
      "addc.cc.u64 %0, %2, %0;"
      : "+l"(res.hi), "+l"(res.lo)
      : "l"(op1.hi), "l"(op1.lo));
}

__inline__ __device__ uint64_t barret_reduction_128_64(
    const uint128_t in,
    const uint64_t prime,
    const uint64_t barret_ratio,
    const uint64_t barret_k) {
  uint128_t temp1 = mult_64_64_128(in.lo, barret_ratio);
  uint128_t temp2 = mult_64_64_128(in.hi, barret_ratio);
  asm("add.cc.u64 %0, %0, %1;" : "+l"(temp1.hi) : "l"(temp2.lo));
  asm("{addc.cc.u64 %0, %0, %1;}" : "+l"(temp2.hi) : "l"((unsigned long)0));
  temp1.hi >>= barret_k - 64;
  temp2.hi <<= 128 - barret_k;
  temp1.hi = temp1.hi + temp2.hi;
  temp1.hi = temp1.hi * prime;
  uint64_t res = in.lo - temp1.hi;
  if (res >= prime)
    res -= prime;
  return res;
}

__inline__ __device__ void barret_reduction_64_64(
    const uint64_t in,
    uint64_t& res,
    const uint64_t prime,
    const uint64_t ratio,
    const uint64_t k) {
  uint64_t hi = __umul64hi(in, ratio);
  hi >>= (k - 64);
  res = in - (hi * prime);
  if (res >= prime)
    res -= prime;
}

__inline__ __device__ uint64_t mul_and_reduce_shoup(
    const uint64_t op1,
    const uint64_t op2,
    const uint64_t scaled_op2,
    const uint64_t prime) {
  uint64_t hi = __umul64hi(scaled_op2, op1);
  return (uint64_t)op1 * op2 - hi * prime;
};



} // namespace fhe