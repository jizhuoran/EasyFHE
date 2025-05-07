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
    uint64_t* out,
    const uint64_t* in,
    const int64_t l,
    const int64_t N,
    const int* precomp_vec) {
  auto tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  auto precomp_index = precomp_vec[tid];
  for (int i = 0; i < l; i++) {
    out[i * N + tid] = in[i * N + precomp_index];
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
  dim3 block(BLOCK_SIZE);
  dim3 grid(N / BLOCK_SIZE);
  fhe::automorphism_transform_kernel<<<grid, block>>>(
      out_ptr, in_ptr, l, N, precomp_vec_ptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

Tensor automorphism_transform_cuda(
    const Tensor& a,
    int64_t l,
    int64_t N,
    const Tensor& precomp_vec) {
  Tensor out = at::empty_like(a);
  automorphism_transform_template(out, a, l, N, precomp_vec);
  return out;
}

} // namespace at::native