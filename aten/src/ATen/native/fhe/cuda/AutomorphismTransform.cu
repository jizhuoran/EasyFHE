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



namespace fhe {
template <size_t NUM_CV>
__global__ void automorphism_transform_kernel(
    uint64_t* out,
    const uint64_t* in,
    const size_t l,
    const size_t N,
    const size_t LN,
    const size_t BLN,
    const int* precomp_vec) {
  auto tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  auto precomp_index = precomp_vec[tid];
  in += blockIdx.y * LN;
  out += blockIdx.y * LN;

#pragma unroll
  for (int cv = 0; cv < NUM_CV; cv++) {
    for (int i = 0; i < l; i++) {
      out[cv * BLN + i * N + tid] = in[cv * BLN + i * N + precomp_index];
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
  dim3 block(BLOCK_SIZE);
  dim3 grid(N / BLOCK_SIZE, out.sizes()[1]);

  auto LN = in.sizes()[2] * N;
  auto BLN = in.sizes()[1] * LN;

  if (out.sizes()[0] == 1) {
    fhe::automorphism_transform_kernel<1>
        <<<grid, block>>>(out_ptr, in_ptr, l, N, LN, BLN, precomp_vec_ptr);
  } else if (out.sizes()[0] == 2) {
    fhe::automorphism_transform_kernel<2>
        <<<grid, block>>>(out_ptr, in_ptr, l, N, LN, BLN, precomp_vec_ptr);
  } else {
    TORCH_INTERNAL_ASSERT(false, "Unsupported number of cv");
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

Tensor automorphism_transform_cuda(
    const Tensor& from,
    int64_t l,
    int64_t N,
    const Tensor& precomp_vec) {
  TORCH_INTERNAL_ASSERT(from.dim() == 4);

  Tensor out = at::empty_like(from);
  automorphism_transform_template(out, from, l, N, precomp_vec);
  return out;
}

} // namespace at::native
