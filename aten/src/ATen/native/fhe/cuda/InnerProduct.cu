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
__global__ void sum_reduce_fused(
    const uint64_t* in_ptr,
    const int N,
    const int length,
    const int mult_length,
    const int batch,
    const uint64_t* eval_ax,
    const uint64_t* eval_bx,
    const uint64_t* primes,
    const uint64_t* barret_ks,
    const uint64_t* barret_ratios,
    int curr_limbs,
    int gap,
    uint64_t* out_ax,
    uint64_t* out_bx) {
  const int idx = blockIdx.y;
  const int i = blockIdx.y * N + blockIdx.x * blockDim.x + threadIdx.x;
  const int prime_idx = ((idx >= 0 && idx < curr_limbs) ? 0 : gap);
  uint128_t accum_ax{0, 0};
  uint128_t accum_bx{0, 0};
  for (int batch_idx = 0; batch_idx < batch; batch_idx++) {
    const int stride = N * mult_length * batch_idx;
    const int in_ptr_stride = N * length * batch_idx;
    const uint64_t op1 = in_ptr[in_ptr_stride + i];
    const uint64_t op2_ax = eval_ax[i + N * prime_idx + stride];
    const auto mul_ax = mult_64_64_128(op1, op2_ax);
    inplace_add_128_128(mul_ax, accum_ax);
    const uint64_t op2_bx = eval_bx[i + N * prime_idx + stride];
    const auto mul_bx = mult_64_64_128(op1, op2_bx);
    inplace_add_128_128(mul_bx, accum_bx);
  }
  const auto reduce_prime_idx = idx + prime_idx;

  const auto prime = primes[reduce_prime_idx];
  const auto barret_ratio = barret_ratios[reduce_prime_idx];
  const auto barret_k = barret_ks[reduce_prime_idx];
  const auto res_ax =
      barret_reduction_128_64(accum_ax, prime, barret_ratio, barret_k);
  const auto res_bx =
      barret_reduction_128_64(accum_bx, prime, barret_ratio, barret_k);
  out_ax[i] = res_ax;
  out_bx[i] = res_bx;
}

} // namespace fhe

namespace at::native {
static void innerproduct_template(
    const Tensor& in,
    const Tensor& bx,
    const Tensor& ax,
    int64_t curr_limbs,
    int64_t alpha,
    int64_t L,
    int64_t N,
    const Tensor& primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    const Tensor& workspace,
    Tensor& out) {
  const int beta = int((curr_limbs + alpha - 1) / alpha);
  int64_t sizeQP = primes.numel();
  int64_t sizeP = sizeQP - L;
  const int length = (curr_limbs + sizeP);
  const int mult_length = (L + sizeP);
  int gap = L - curr_limbs;

  auto in_ptr = reinterpret_cast<uint64_t*>(in.data_ptr<uint64_t>());
  auto ax_ptr = reinterpret_cast<uint64_t*>(ax.data_ptr<uint64_t>());
  auto bx_ptr = reinterpret_cast<uint64_t*>(bx.data_ptr<uint64_t>());
  auto out_bx_ptr = reinterpret_cast<uint64_t*>(out[0].data_ptr<uint64_t>());
  auto out_ax_ptr = reinterpret_cast<uint64_t*>(out[1].data_ptr<uint64_t>());
  auto primes_ptr = reinterpret_cast<uint64_t*>(primes.data_ptr<uint64_t>());
  auto barret_ratio_ptr =
      reinterpret_cast<uint64_t*>(barret_ratio.data_ptr<uint64_t>());
  auto barret_k_ptr =
      reinterpret_cast<uint64_t*>(barret_k.data_ptr<uint64_t>());
  auto gridDim = dim3(N / 256, length);
  auto blockDim = 256;
  auto stream = at::cuda::getCurrentCUDAStream();
  fhe::sum_reduce_fused<<<gridDim, blockDim, 0, stream>>>(
      in_ptr,
      N,
      length,
      mult_length,
      beta,
      ax_ptr,
      bx_ptr,
      primes_ptr,
      barret_k_ptr,
      barret_ratio_ptr,
      curr_limbs,
      gap,
      out_ax_ptr,
      out_bx_ptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

Tensor innerproduct_cuda(
    const Tensor& res,
    const Tensor& in,
    const Tensor& bx,
    const Tensor& ax,
    int64_t curr_limbs,
    int64_t alpha,
    int64_t L,
    int64_t N,
    const Tensor& primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    const Tensor& workspace) {
  Tensor out = at::empty_like(res);
  int64_t sizeQP = primes.numel();
  int64_t sizeP = sizeQP - L;
  out.resize_({2, (curr_limbs + sizeP) * N});
  innerproduct_template(
      in,
      bx,
      ax,
      curr_limbs,
      alpha,
      L,
      N,
      primes,
      barret_ratio,
      barret_k,
      workspace,
      out);
  return out;
}

} // namespace at::native