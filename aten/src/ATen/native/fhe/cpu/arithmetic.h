#pragma once

#include <cstdint>
namespace fhe {

static inline __attribute__((always_inline)) uint64_t
neg_mod(uint64_t x, uint64_t null, uint64_t mod) {
  return x == 0 ? 0 : mod - x;
}

static inline __attribute__((always_inline)) uint64_t
add_mod(uint64_t a, uint64_t b, uint64_t mod) {
  uint64_t res = a + b;
  return res >= mod ? res - mod : res;
}

static inline __attribute__((always_inline)) uint64_t
sub_mod(uint64_t a, uint64_t b, uint64_t mod) {
  return a >= b ? a - b : a + mod - b;
}

static inline __attribute__((always_inline)) uint64_t
mul_mod(uint64_t a, uint64_t b, uint64_t mod, uint64_t mu0, uint64_t mu1) {
  unsigned __int128 p = (unsigned __int128)a * b;
  uint64_t p_lo = (uint64_t)p;
  uint64_t p_hi = p >> 64;

  unsigned __int128 t = ((unsigned __int128)p_lo * mu0) >> 64;
  t += (unsigned __int128)p_lo * mu1;
  t += (unsigned __int128)p_hi * mu0;

  uint64_t q = (uint64_t)(t >> 64) + (uint64_t)((unsigned __int128)p_hi * mu1);

  unsigned __int128 r = p - (unsigned __int128)q * mod;

  return (uint64_t)(r >= mod ? r - mod : r);
}

} // namespace fhe