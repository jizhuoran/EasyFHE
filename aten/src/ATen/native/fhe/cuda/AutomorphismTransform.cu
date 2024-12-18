#include <ATen/Dispatch_v2.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/interned_strings.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/thread_constants.h>
#include <ATen/ops/copy.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/stack.h>
#include <ATen/ops/zeros.h>

#include "ATen/native/fhe/cuda/Utils.cuh"

#define WORK_PER_THREAD (1)
#define WARP_SIZE (32)
#define NUM_WARPS (8)
#define BLOCK_SIZE (WARP_SIZE * NUM_WARPS)
#define WORK_PER_BLOCK (WORK_PER_THREAD * BLOCK_SIZE)

#define num_blocks(n) ((n + WORK_PER_BLOCK - 1) / WORK_PER_BLOCK)

namespace fhe {
__global__ void automorphism_transform_kernel(
    uint64_t* ra,
    const uint64_t* a,
    int l,
    int N,
    int i,
    const int* precomp_vec) {
  STRIDED_LOOP_START(l * N, idx)
  int k = idx / N; // Index for the segment in l
  int j = idx % N; // Index for the elements within the segment (up to N)}
  if (k < l && j < N) {
    int offset = k * N;
    ra[idx] = a[offset + precomp_vec[j]];
  }
  STRIDED_LOOP_END;
}
} // namespace fhe

namespace at::native {
static void automorphism_transform_template(
    Tensor& ra,
    const Tensor& a,
    int64_t l,
    int64_t N,
    int64_t i,
    const Tensor& precomp_vec) {
  if (i % 2 == 0) {
    return;
  }

  AT_DISPATCH_V2(
      a.scalar_type(),
      "automorphism_transform_impl",
      AT_WRAP([&]() {
        auto ra_ptr = reinterpret_cast<uint64_t*>(ra.data_ptr<uint64_t>());
        auto a_ptr = reinterpret_cast<uint64_t*>(a.data_ptr<uint64_t>());
        auto precomp_vec_ptr =
            reinterpret_cast<int32_t*>(precomp_vec.data_ptr<int32_t>());
        const int block_dim = 256;
        const int grid_dim = N * l / block_dim;
        auto stream = at::cuda::getCurrentCUDAStream();
        fhe::automorphism_transform_kernel<<<grid_dim, block_dim, 0, stream>>>(
            ra_ptr, a_ptr, l, N, i, precomp_vec_ptr);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }),
      kUInt64);
}

Tensor automorphism_transform_cuda(
    const Tensor& ra,
    const Tensor& a,
    int64_t l,
    int64_t N,
    int64_t i,
    const Tensor& precomp_vec) {
  Tensor out = at::empty_like(a);
  automorphism_transform_template(out, a, l, N, i, precomp_vec);
  return out;
}

Tensor& automorphism_transform_cuda_(
    Tensor& ra,
    const Tensor& a,
    int64_t l,
    int64_t N,
    int64_t i,
    const Tensor& precomp_vec) {
  ra.resize_({l * N});
  automorphism_transform_template(ra, a, l, N, i, precomp_vec);
  return ra;
}

Tensor& automorphism_transform_cuda_out(
    const Tensor& ra,
    const Tensor& a,
    int64_t l,
    int64_t N,
    int64_t i,
    const Tensor& precomp_vec,
    Tensor& out) {
  ra.resize_({l * N});
  automorphism_transform_template(out, a, l, N, i, precomp_vec);
  return out;
}

} // namespace at::native