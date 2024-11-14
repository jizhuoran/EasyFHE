#include <ATen/Dispatch_v2.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/interned_strings.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/thread_constants.h>
#include <ATen/ops/copy.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>

#include "ATen/native/fhe/cuda/KeySwitch.h"

#pragma clang diagnostic ignored "-Wmissing-prototypes"

#define WORK_PER_THREAD (1)
#define WARP_SIZE (32)
#define NUM_WARPS (8)
#define BLOCK_SIZE (WARP_SIZE * NUM_WARPS)
#define WORK_PER_BLOCK (WORK_PER_THREAD * BLOCK_SIZE)

#define num_blocks(n) ((n + WORK_PER_BLOCK - 1) / WORK_PER_BLOCK)

namespace fhe {
__global__ void mulByMonomialKernel(
    uint64_t* res,
    uint64_t* qVec,
    uint64_t* tmp,
    long l,
    long N,
    long shift) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < l * N) {
    long i = idx / N; // 确定当前处理的是哪个 limb
    long n = idx % N; // 确定 limb 中的元素索引

    // 从 a 复制到 tmp 或计算负数情况
    if (shift < N) {
      tmp[idx] = res[idx];
    } else {
      tmp[idx] = qVec[i] - res[idx];
    }

    // 执行 shift 操作
    if (n < shift) {
      res[idx] = qVec[i] - tmp[(i * N) + (N - shift + n)];
    } else {
      res[idx] = tmp[(i * N) + (n - shift)];
    }
  }
}
} // namespace fhe

namespace at::native {

static void mul_by_monomial_impl(
    Tensor& res,
    const Tensor& qVec,
    const Tensor& tmp,
    int64_t l,
    int64_t N,
    int64_t M,
    int64_t monomialDeg) {
  int64_t shift = monomialDeg % M;

  AT_DISPATCH_V2(
      res.scalar_type(),
      "mul_by_monomial_impl",
      AT_WRAP([&]() {
        auto res_ptr = reinterpret_cast<uint64_t*>(res.data_ptr<uint64_t>());
        auto qvec_ptr = reinterpret_cast<uint64_t*>(qVec.data_ptr<uint64_t>());
        auto tmp_ptr = reinterpret_cast<uint64_t*>(tmp.data_ptr<uint64_t>());
        int blockDim = 256; // One block for each `k` (segment)
        int gridDim = (l * N + blockDim - 1) /
            blockDim; // One thread for each element in a segment (up to `N`)
        auto stream = at::cuda::getCurrentCUDAStream();
        fhe::mulByMonomialKernel<<<gridDim, blockDim, 0, stream>>>(
            res_ptr, qvec_ptr, tmp_ptr, l, N, shift);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }),
      kUInt64);
}

static void mul_by_monomial_template(
    Tensor& res,
    const Tensor& qVec,
    const Tensor& tmp,
    const Tensor& param_primes,
    int64_t l,
    int64_t N,
    int64_t M,
    int64_t monomialDeg,
    int64_t curr_limbs,
    int64_t level,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_power_of_roots) {
  int64_t shift = monomialDeg % M;
  auto res_ptr = reinterpret_cast<uint64_t*>(res.data_ptr<uint64_t>());
  iNTT_impl(
      res_ptr,
      0,
      curr_limbs,
      curr_limbs,
      level,
      N,
      inverse_power_of_roots_div_two,
      param_primes,
      inverse_scaled_power_of_roots_div_two);

  mul_by_monomial_impl(res, qVec, tmp, l, N, M, monomialDeg);

      NTT_impl(
          res_ptr,
          0,
          curr_limbs,
          N,
          param_power_of_roots_shoup,
          param_primes,
          param_power_of_roots);
}

Tensor mul_by_monomial_cuda(
    const Tensor& res,
    const Tensor& qVec,
    const Tensor& tmp,
    const Tensor& param_primes,
    int64_t l,
    int64_t N,
    int64_t M,
    int64_t monomialDeg,
    int64_t curr_limbs,
    int64_t level,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_power_of_roots) {
  Tensor out = res.clone();
  mul_by_monomial_template(
      out,
      qVec,
      tmp,
      param_primes,
      l,
      N,
      M,
      monomialDeg,
      curr_limbs,
      level,
      inverse_power_of_roots_div_two,
      inverse_scaled_power_of_roots_div_two,
      param_power_of_roots_shoup,
      param_power_of_roots);
  return out;
}

Tensor& mul_by_monomial_cuda_(
    Tensor& res,
    const Tensor& qVec,
    const Tensor& tmp,
    const Tensor& param_primes,
    int64_t l,
    int64_t N,
    int64_t M,
    int64_t monomialDeg,
    int64_t curr_limbs,
    int64_t level,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_power_of_roots) {
  mul_by_monomial_template(
      res,
      qVec,
      tmp,
      param_primes,
      l,
      N,
      M,
      monomialDeg,
      curr_limbs,
      level,
      inverse_power_of_roots_div_two,
      inverse_scaled_power_of_roots_div_two,
      param_power_of_roots_shoup,
      param_power_of_roots);
  return res;
}

Tensor& mul_by_monomial_cuda_out(
    const Tensor& res,
    const Tensor& qVec,
    const Tensor& tmp,
    const Tensor& param_primes,
    int64_t l,
    int64_t N,
    int64_t M,
    int64_t monomialDeg,
    int64_t curr_limbs,
    int64_t level,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_power_of_roots,
    Tensor& out) {
  mul_by_monomial_template(
      out,
      qVec,
      tmp,
      param_primes,
      l,
      N,
      M,
      monomialDeg,
      curr_limbs,
      level,
      inverse_power_of_roots_div_two,
      inverse_scaled_power_of_roots_div_two,
      param_power_of_roots_shoup,
      param_power_of_roots);
  return out;
}

} // namespace at::native