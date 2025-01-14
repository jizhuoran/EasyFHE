#pragma once

#include <cstdint>

namespace fhe {


static inline __attribute__((always_inline))
uint64_t neg_mod(uint64_t x, uint64_t null, uint64_t mod) {
  return x == 0 ? 0 : mod - x;
}


static inline __attribute__((always_inline))
uint64_t add_mod(uint64_t a, uint64_t b, uint64_t mod) {
  uint64_t res = a + b;
  return res >= mod ? res - mod : res;
}


static inline __attribute__((always_inline))
uint64_t sub_mod(uint64_t a, uint64_t b, uint64_t mod) {
  return a >= b ? a - b : a + mod - b;
}


static inline __attribute__((always_inline))
uint64_t mul_mod(uint64_t a,
                     uint64_t b,
                     uint64_t mod,
                     uint64_t barret_mu0,
                     uint64_t barret_mu1)
{
    uint64_t lo, hi, tmp, res;

    // The structure below follows the same sequence as the PTX:
    // 1) 128-bit multiply of (a, b)
    // 2) Multiply pieces by barret_mu0/barret_mu1
    // 3) Sums with carry (mad.lo, madc.hi)
    // 4) Final multiply and subtract for the Barrett step
    // 5) Compare with mod to finalize.

    asm volatile(
        // -------------------------------------------------------
        // 128-bit multiply: (a * b) => hi:lo
        // -------------------------------------------------------
        "movq  %[a],  %%rax        \n\t"
        "mulq  %[b]               \n\t"  // (RDX:RAX) = a*b
        "movq  %%rax, %[lo]       \n\t"  // lo = RAX
        "movq  %%rdx, %[hi]       \n\t"  // hi = RDX

        // -------------------------------------------------------
        // tmp = hi( lo * barret_mu0 ) after 64-bit multiplication
        // -------------------------------------------------------
        "movq  %[lo], %%rax       \n\t"
        "mulq  %[barret_mu0]      \n\t"  // (RDX:RAX) = lo * barret_mu0
        "movq  %%rdx, %[tmp]      \n\t"  // tmp = high part

        // -------------------------------------------------------
        // tmp += (lo * barret_mu1).LO (lowest 64 bits),
        // and res = carry + (lo * barret_mu1).HI
        // This mimics mad.lo.cc.u64 + madc.hi.u64
        // -------------------------------------------------------
        "movq  %[lo], %%rax       \n\t"
        "mulq  %[barret_mu1]      \n\t"  // (RDX:RAX) = lo * barret_mu1
        "addq  %%rax, %[tmp]      \n\t"  // tmp += low part
        "movq  $0,     %[res]     \n\t"  // Initialize res = 0
        "adcq  %%rdx, %[res]      \n\t"  // res = carry + high part

        // -------------------------------------------------------
        // Round 2: same pattern but with hi * barret_mu0
        // tmp += (hi * barret_mu0).LO
        // res += carry + (hi * barret_mu0).HI
        // -------------------------------------------------------
        "movq  %[hi], %%rax       \n\t"
        "mulq  %[barret_mu0]      \n\t"  // (RDX:RAX) = hi * barret_mu0
        "addq  %%rax, %[tmp]      \n\t"  // tmp += low part
        "adcq  %%rdx, %[res]      \n\t"  // res += carry + high part

        // -------------------------------------------------------
        // This is "mad.lo.u64 res, hi, barret_mu1, res":
        // res += (hi * barret_mu1).LO
        // ignoring carry out
        // -------------------------------------------------------
        "movq  %[hi], %%rax       \n\t"
        "mulq  %[barret_mu1]      \n\t"  // (RDX:RAX) = hi * barret_mu1
        "addq  %%rax, %[res]      \n\t"  // res += low part (no carry handling)

        // -------------------------------------------------------
        // Barrett subtraction:
        // res = lo - (res * mod).LO
        // -------------------------------------------------------
        "movq  %[res], %%rax      \n\t"
        "mulq  %[mod]             \n\t"  // (RDX:RAX) = res * mod
        "subq  %%rax, %[lo]       \n\t"  // lo = lo - (res * mod).LO
        "movq  %[lo], %[res]      \n\t"  // res = updated lo

        : [lo]  "=&r" (lo),
          [hi]  "=&r" (hi),
          [tmp] "=&r" (tmp),
          [res] "=&r" (res)
        : [a]           "r" (a),
          [b]           "r" (b),
          [mod]         "r" (mod),
          [barret_mu0]  "r" (barret_mu0),
          [barret_mu1]  "r" (barret_mu1)
        : "rax", "rdx", "cc"
    );

    // Final conditional reduction: if (res >= mod) res -= mod
    if (res >= mod) {
        res -= mod;
    }
    return res;
}

} // namespace fhe