#include <ATen/Dispatch_v2.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/interned_strings.h>
#include <ATen/ops/copy.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/stack.h>
#include <ATen/ops/zeros.h>

#include "ATen/native/fhe/cpu/Utils.h"

#define WORK_PER_THREAD (1)
#define WARP_SIZE (32)
#define NUM_WARPS (8)
#define BLOCK_SIZE (WARP_SIZE * NUM_WARPS)
#define WORK_PER_BLOCK (WORK_PER_THREAD * BLOCK_SIZE)

#define num_blocks(n) ((n + WORK_PER_BLOCK - 1) / WORK_PER_BLOCK)

namespace fhe {
void automorphism_transform_kernel(
    uint64_t* out,
    const uint64_t* in,
    const int l,
    const int N,
    const int* precomp_vec) {
  for (int j = 0; j < l; j++) {
    for (int i = 0; i < N; i++) {
      out[j * N + i] = in[j * N + precomp_vec[i]];
    }
  }
}

} // namespace fhe

namespace at::native {
static void automorphism_transform_template(
    Tensor& out,
    const Tensor& in,
    int64_t l,
    int64_t N,
    const Tensor& precomp_vec) {
  auto out_ptr = reinterpret_cast<uint64_t*>(out.data_ptr<uint64_t>());
  auto in_ptr = reinterpret_cast<uint64_t*>(in.data_ptr<uint64_t>());
  auto precomp_vec_ptr =
      reinterpret_cast<int32_t*>(precomp_vec.data_ptr<int32_t>());

  fhe::automorphism_transform_kernel(out_ptr, in_ptr, l, N, precomp_vec_ptr);
}

Tensor automorphism_transform_cpu(
    const Tensor& a,
    int64_t l,
    int64_t N,
    const Tensor& precomp_vec) {
  Tensor out = at::empty_like(a);
  automorphism_transform_template(out, a, l, N, precomp_vec);
  return out;
}

Tensor& automorphism_transform_cpu_(
    Tensor& a,
    int64_t l,
    int64_t N,
    const Tensor& precomp_vec) {
  automorphism_transform_template(a, a, l, N, precomp_vec);
  return a;
}

Tensor& automorphism_transform_cpu_out(
    const Tensor& a,
    int64_t l,
    int64_t N,
    const Tensor& precomp_vec,
    Tensor& out) {
  automorphism_transform_template(out, a, l, N, precomp_vec);
  return out;
}

} // namespace at::native