#pragma once

namespace fhe {

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
      " .reg .u64 tmp;\n\t"
      " .reg .u64 lo, hi;\n\t"
      // 128-bit multiply
      " mul.lo.u64 lo, %1, %2;\n\t"
      " mul.hi.u64 hi, %1, %2;\n\t"
      // Multiply input and const_ratio
      // Round 1
      " mul.hi.u64 tmp, lo, %3;\n\t"
      " mad.lo.cc.u64 tmp, lo, %4, tmp;\n\t"
      " madc.hi.u64 %0, lo, %4, 0;\n\t"
      // Round 2
      " mad.lo.cc.u64 tmp, hi, %3, tmp;\n\t"
      " madc.hi.u64 %0, hi, %3, %0;\n\t"
      // This is all we care about
      " mad.lo.u64 %0, hi, %4, %0;\n\t"
      // Barrett subtraction
      " mul.lo.u64 %0, %0, %5;\n\t"
      " sub.u64 %0, lo, %0;\n\t"
      "}"
      : "=l"(res)
      : "l"(a), "l"(b), "l"(barret_mu0), "l"(barret_mu1), "l"(mod));
  return res >= mod ? res - mod : res;
}

} // namespace fhe
