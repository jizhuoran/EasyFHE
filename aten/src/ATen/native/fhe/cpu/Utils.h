#pragma once

#include <cstdint>

#define STRIDED_LOOP_START(N, i) for (int i = 0; i < N; i++) {
#define STRIDED_LOOP_END }
#define TO_PTR(x) (x.data())

static inline __attribute__((always_inline))
void __syncthreads() {
  // Dummy implementation
}

namespace fhe {

using uint128_t = __uint128_t;

struct uint128_t4 {
  uint128_t x;
  uint128_t y;
  uint128_t z;
  uint128_t w;
};

inline uint64_t umul64hi_inline_asm(uint64_t a, uint64_t b) {
  // After `mulq b`, RDX:RAX = a * b, so the high 64 bits end up in RDX.
  uint64_t hi;
  asm("mulq %2"
      : "=d"(hi), // RDX
        "+a"(a) // RAX is both an input (a) and updated with the low part
      : "r"(b)
      : "cc");
  return hi;
}


static inline void inplace_add_128_128(const uint128_t op1, uint128_t &res) {
  res += op1;
}

static inline uint128_t mult_64_64_128(uint64_t a, uint64_t b)
{
  return (uint128_t)a * b;
}

static inline uint64_t barret_reduction_128_64(
    const uint128_t in,
    const uint64_t prime,
    const uint64_t barret_ratio,
    const uint64_t barret_k)
{
  uint64_t _in_lo = (uint64_t)(in & 0xFFFFFFFFFFFFFFFF);
  uint64_t _in_hi = (uint64_t)((in >> 64) & 0xFFFFFFFFFFFFFFFF);

  uint128_t temp1 = mult_64_64_128(_in_lo, barret_ratio);
  uint128_t temp2 = mult_64_64_128(_in_hi, barret_ratio);

  uint64_t _temp1_hi = (uint64_t)((temp1 >> 64) & 0xFFFFFFFFFFFFFFFF);
  uint64_t _temp2_hi = (uint64_t)((temp2 >> 64) & 0xFFFFFFFFFFFFFFFF);
  uint64_t _temp1_lo = (uint64_t)(temp1 & 0xFFFFFFFFFFFFFFFF);
  uint64_t _temp2_lo = (uint64_t)(temp2 & 0xFFFFFFFFFFFFFFFF);

  _temp1_hi += _temp2_lo;
  if (_temp1_hi < _temp2_lo) {
    _temp2_hi++;
  }

  // 3) Shifts
  _temp1_hi >>= (barret_k - 64);
  _temp2_hi <<= (128 - barret_k);

  _temp1_hi += _temp2_hi;
  _temp1_hi *= prime;

  // 4) Final subtraction mod prime
  uint64_t res = _in_lo - _temp1_hi;
  if (res >= prime) {
      res -= prime;
  }
  return res;
}


static inline uint32_t umulhi_32_32(uint32_t a, uint32_t b)
{
    // EAX := a;  mul b => EDX:EAX = a*b (64-bit result)
    // We take EDX (the high 32 bits).
    uint32_t hi;
    __asm__ (
        "mull %[b]"
        : "=d" (hi),    // EDX
          "+a" (a)      // EAX in/out
        : [b] "r" (b)
        : "cc"
    );
    return hi;
}


static inline uint32_t barret_reduction_64_32(
    const uint64_t in,
    const uint32_t prime,
    const uint32_t barret_ratio,
    const uint64_t barret_k)
{
    // 1) Split ‘in’ into low/high 32 bits
    uint32_t in_lo = (uint32_t) in;
    uint32_t in_hi = (uint32_t)(in >> 32);

    // 2) Multiply to get partial products
    uint32_t temp1_hi = umulhi_32_32(in_lo, barret_ratio);
    uint64_t temp2    = (uint64_t)in_hi * barret_ratio;

    // 3) Break temp2 into hi/lo 32 bits
    uint32_t temp2_hi = (uint32_t)(temp2 >> 32);
    uint32_t temp2_lo = (uint32_t) temp2;

    // 4) Add with carry
    asm volatile(
        "addl  %[temp2_lo], %[temp1_hi] \n\t" // (temp1_hi += temp2_lo)
        "adcl  $0,        %[temp2_hi]   \n\t" // (temp2_hi += carry)
        : [temp1_hi] "+r" (temp1_hi),
          [temp2_hi] "+r" (temp2_hi)
        : [temp2_lo] "r" (temp2_lo)
        : "cc"
    );

    // 5) Shifts
    temp1_hi >>= (barret_k - 32);
    temp2_hi <<= (64 - barret_k);

    temp1_hi += temp2_hi;
    temp1_hi *= prime;

    // 6) Final reduce
    uint32_t res = in_lo - temp1_hi;
    if (res >= prime) {
        res -= prime;
    }
    return res;
}

__inline__ void barret_reduction_64_64(
    const uint64_t in,
    uint64_t& res,
    const uint64_t prime,
    const uint64_t ratio,
    const uint64_t k) {
  uint64_t hi = umul64hi_inline_asm(in, ratio);
  hi >>= (k - 64);
  res = in - (hi * prime);
  if (res >= prime)
    res -= prime;
}

__inline__ uint64_t mul_and_reduce_shoup(
    const uint64_t op1,
    const uint64_t op2,
    const uint64_t scaled_op2,
    const uint64_t prime) {
  uint64_t hi = umul64hi_inline_asm(scaled_op2, op1);
  return (uint64_t)op1 * op2 - hi * prime;
};

__inline__ uint64_t sub_negate_const_mult(
    const uint64_t op1,
    const uint64_t op2,
    const uint64_t op3,
    const uint64_t scaled_op3,
    const uint64_t prime) {
  uint64_t temp;
  if (op1 >= op2)
    temp = prime - op1 + op2;
  else {
    temp = op2 - op1;
  }
  uint64_t out = mul_and_reduce_shoup(temp, op3, scaled_op3, prime);
  if (out >= prime)
    out -= prime;
  return out;
};


} // namespace fhe