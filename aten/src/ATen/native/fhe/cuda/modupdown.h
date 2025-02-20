#pragma once
namespace fhe {

__inline__ __device__ uint128_t4 accumulate_in_modupdown(
  const uint64_t* ptr,
  const int N,
  const uint64_t* hat_mod_end,
  const int start_length,
  const int degree_idx,
  const int hat_mod_end_idx) {
uint128_t4 accum{0};
for (int i = 0; i < start_length; i++) {
  const uint64_t op2 = hat_mod_end[hat_mod_end_idx * start_length + i];
  uint128_t4 out;
ulonglong4 op1;
op1 = *reinterpret_cast<const ulonglong4*>(ptr + i * N + degree_idx);

  out.x = mult_64_64_128(op1.x, op2);
  inplace_add_128_128(out.x, accum.x);
  out.y = mult_64_64_128(op1.y, op2);
  inplace_add_128_128(out.y, accum.y);
  out.z = mult_64_64_128(op1.z, op2);
  inplace_add_128_128(out.z, accum.z);
  out.w = mult_64_64_128(op1.w, op2);
  inplace_add_128_128(out.w, accum.w);
}
return accum;
}

} //namespace fhe

namespace at::native {

const int unroll_factor = 4;

} //namespace at::native