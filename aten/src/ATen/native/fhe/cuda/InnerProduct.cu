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
__global__ void sumAndReduceFused(
    const uint64_t* modup_out,
    const int degree,
    const int length,
    const int batch,
    const uint64_t* eval_ax,
    const uint64_t* eval_bx,
    const uint64_t* primes,
    const uint64_t* barret_ks,
    const uint64_t* barret_ratios,
    uint64_t* dst_ax,
    uint64_t* dst_bx) {
  STRIDED_LOOP_START(degree * length, i);
  const int stride_between_batch = degree * length;
  uint128_t accum_ax{0, 0};
  uint128_t accum_bx{0, 0};
  for (int batch_idx = 0; batch_idx < batch; batch_idx++) {
    const int idx = i + stride_between_batch * batch_idx;
    const uint64_t op1 = modup_out[idx];
    const uint64_t op2_ax = eval_ax[idx];
    const auto mul_ax = mult_64_64_128(op1, op2_ax);
    accum_ax += mul_ax;
    const uint64_t op2_bx = eval_bx[idx];
    const auto mul_bx = mult_64_64_128(op1, op2_bx);
    accum_bx += mul_bx;
  }
  const int prime_idx = i / degree;
  const auto prime = primes[prime_idx];
  const auto barret_ratio = barret_ratios[prime_idx];
  const auto barret_k = barret_ks[prime_idx];
  const auto res_ax =
      barret_reduction_128_64(accum_ax, prime, barret_ratio, barret_k);
  const auto res_bx =
      barret_reduction_128_64(accum_bx, prime, barret_ratio, barret_k);
  dst_ax[i] = res_ax;
  dst_bx[i] = res_bx;
  STRIDED_LOOP_END;
}

template <bool Accum>
__global__ void mult_(
    const uint64_t* modup_out,
    const uint64_t* eval_poly_ax,
    const uint64_t* eval_poly_bx,
    const int degree,
    const int length,
    uint128_t* accum_ptr_ax,
    uint128_t* accum_ptr_bx,
    int curr_limbs,
    int gap) {
  STRIDED_LOOP_START(degree * length, i);
  const uint64_t op1 = modup_out[i];
  const int idx = i / degree;
  const int prime_idx = ((idx >= 0 && idx < curr_limbs) ? 0 : gap);

  const uint64_t op2_ax = eval_poly_ax[i + degree * prime_idx];
  const uint64_t op2_bx = eval_poly_bx[i + degree * prime_idx];
  const auto mul_ax = mult_64_64_128(op1, op2_ax);
  const auto mul_bx = mult_64_64_128(op1, op2_bx);
  if (Accum) {
    accum_ptr_ax[i] += mul_ax;
    accum_ptr_bx[i] += mul_bx;
  } else {
    accum_ptr_ax[i] = mul_ax;
    accum_ptr_bx[i] = mul_bx;
  }
  STRIDED_LOOP_END;
}

__global__ void Reduce(
    const uint128_t* accum,
    const int degree,
    const int length,
    const int curr_limbs,
    const int gap,
    const uint64_t* primes,
    const uint64_t* barret_ks,
    const uint64_t* barret_ratios,
    uint64_t* res) {
  STRIDED_LOOP_START(degree * length, i);
  const int idx = i / degree;
  const int prime_idx = idx + ((idx >= 0 && idx < curr_limbs) ? 0 : gap);
  const auto prime = primes[prime_idx];
  const auto barret_ratio = barret_ratios[prime_idx];
  const auto barret_k = barret_ks[prime_idx];
  const auto res_ax =
      barret_reduction_128_64(accum[i], prime, barret_ratio, barret_k);
  res[i] = res_ax;
  STRIDED_LOOP_END;
}
} // namespace fhe

namespace at::native {
static void innerproduct_template(
    const Tensor& modup_out,
    const Tensor& ax,
    const Tensor& bx,
    int64_t curr_limbs,
    int64_t alpha,
    int64_t level,
    int64_t param_degree,
    const Tensor& primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    const Tensor& workspace,
    Tensor& res) {
  const int total_length = modup_out.size(-1) / param_degree;
  const int beta = total_length / (curr_limbs + alpha);
  const int length = (curr_limbs + alpha);
  const int mult_length = (level + alpha);
  int gap = level - curr_limbs;

  fhe::uint128_t* accum_ax_ptr =
      reinterpret_cast<fhe::uint128_t*>(workspace.data_ptr<uint64_t>());
  fhe::uint128_t* accum_bx_ptr = accum_ax_ptr + modup_out.size(-1);

  AT_DISPATCH_V2(
      ax.scalar_type(),
      "inner_product_impl",
      AT_WRAP([&]() {
        auto modup_out_ptr =
            reinterpret_cast<uint64_t*>(modup_out.data_ptr<uint64_t>());
        auto ax_ptr = reinterpret_cast<uint64_t*>(ax.data_ptr<uint64_t>());
        auto bx_ptr = reinterpret_cast<uint64_t*>(bx.data_ptr<uint64_t>());
        auto res_ax_ptr =
            reinterpret_cast<uint64_t*>(res[0].data_ptr<uint64_t>());
        auto res_bx_ptr =
            reinterpret_cast<uint64_t*>(res[1].data_ptr<uint64_t>());
        auto primes_ptr =
            reinterpret_cast<uint64_t*>(primes.data_ptr<uint64_t>());
        auto barret_ratio_ptr =
            reinterpret_cast<uint64_t*>(barret_ratio.data_ptr<uint64_t>());
        auto barret_k_ptr =
            reinterpret_cast<uint64_t*>(barret_k.data_ptr<uint64_t>());
        const int gridDim = 1024;
        const int blockDim = 256;
        auto stream = at::cuda::getCurrentCUDAStream();
        fhe::mult_<false><<<gridDim, blockDim, 0, stream>>>(
            modup_out_ptr,
            ax_ptr,
            bx_ptr,
            param_degree,
            length,
            accum_ax_ptr,
            accum_bx_ptr,
            curr_limbs,
            gap);
        for (int i = 1; i < beta; i++) {
          auto d2_ptr = modup_out_ptr + i * param_degree * length;
          auto d_ax_ptr = ax_ptr + i * param_degree * mult_length;
          auto d_bx_ptr = bx_ptr + i * param_degree * mult_length;
          fhe::mult_<true><<<gridDim, blockDim, 0, stream>>>(
              d2_ptr,
              d_ax_ptr,
              d_bx_ptr,
              param_degree,
              length,
              accum_ax_ptr,
              accum_bx_ptr,
              curr_limbs,
              gap);
        }
        fhe::Reduce<<<gridDim, blockDim, 0, stream>>>(
            accum_ax_ptr,
            param_degree,
            length,
            curr_limbs,
            gap,
            primes_ptr,
            barret_k_ptr,
            barret_ratio_ptr,
            res_ax_ptr);
        fhe::Reduce<<<gridDim, blockDim, 0, stream>>>(
            accum_bx_ptr,
            param_degree,
            length,
            curr_limbs,
            gap,
            primes_ptr,
            barret_k_ptr,
            barret_ratio_ptr,
            res_bx_ptr);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }),
      kUInt64);
}

Tensor innerproduct_cuda(
    const Tensor& res,
    const Tensor& modup_out,
    const Tensor& ax,
    const Tensor& bx,
    int64_t curr_limbs,
    int64_t alpha,
    int64_t level,
    int64_t param_degree,
    const Tensor& primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    const Tensor& workspace) {
  Tensor out = at::empty_like(res);
  out.resize_({2, (curr_limbs + alpha) * param_degree});
  innerproduct_template(
      modup_out,
      ax,
      bx,
      curr_limbs,
      alpha,
      level,
      param_degree,
      primes,
      barret_ratio,
      barret_k,
      workspace,
      out);
  return out;
}

Tensor& innerproduct_cuda_(
    Tensor& res,
    const Tensor& modup_out,
    const Tensor& ax,
    const Tensor& bx,
    int64_t curr_limbs,
    int64_t alpha,
    int64_t level,
    int64_t param_degree,
    const Tensor& primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    const Tensor& workspace) {
  innerproduct_template(
      modup_out,
      ax,
      bx,
      curr_limbs,
      alpha,
      level,
      param_degree,
      primes,
      barret_ratio,
      barret_k,
      workspace,
      res);
  return res;
}

Tensor& innerproduct_cuda_out(
    const Tensor& res,
    const Tensor& modup_out,
    const Tensor& ax,
    const Tensor& bx,
    int64_t curr_limbs,
    int64_t alpha,
    int64_t level,
    int64_t param_degree,
    const Tensor& primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    const Tensor& workspace,
    Tensor& out) {
  innerproduct_template(
      modup_out,
      ax,
      bx,
      curr_limbs,
      alpha,
      level,
      param_degree,
      primes,
      barret_ratio,
      barret_k,
      workspace,
      out);
  return out;
}

} // namespace at::native