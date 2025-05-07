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

#include "ATen/native/fhe/cuda/CommonOperation.h"
#include "ATen/native/fhe/cuda/Utils.cuh"

#pragma clang diagnostic ignored "-Wmissing-prototypes"

#define WORK_PER_THREAD (1)
#define WARP_SIZE (32)
#define NUM_WARPS (8)
#define BLOCK_SIZE (WARP_SIZE * NUM_WARPS)
#define WORK_PER_BLOCK (WORK_PER_THREAD * BLOCK_SIZE)

#define num_blocks(n) ((n + WORK_PER_BLOCK - 1) / WORK_PER_BLOCK)

namespace fhe {

#define GEN_MUL(IDX)                               \
  auto ptx_val##IDX = ptx##IDX[tid_y * N + tid_x]; \
  auto tmp_bx##IDX = mul_mod(                      \
      bx##IDX[tid_y * N + tid_x],                  \
      ptx_val##IDX,                                \
      mod_val,                                     \
      barret_mu_val0,                              \
      barret_mu_val1);                             \
  auto tmp_ax##IDX = mul_mod(                      \
      ax##IDX[tid_y * N + tid_x],                  \
      ptx_val##IDX,                                \
      mod_val,                                     \
      barret_mu_val0,                              \
      barret_mu_val1);

#define GEN_SUM(IDX)                              \
  sum_bx = add_mod(sum_bx, tmp_bx##IDX, mod_val); \
  sum_ax = add_mod(sum_ax, tmp_ax##IDX, mod_val)

#define GEN_PARAM(IDX)                                                        \
  const uint64_t* __restrict__ bx##IDX, const uint64_t* __restrict__ ax##IDX, \
      const uint64_t* __restrict__ ptx##IDX

__global__ void fusedPairwiseMACKernel9(
    GEN_PARAM(0),
    GEN_PARAM(1),
    GEN_PARAM(2),
    GEN_PARAM(3),
    GEN_PARAM(4),
    GEN_PARAM(5),
    GEN_PARAM(6),
    GEN_PARAM(7),
    GEN_PARAM(8),
    uint64_t* __restrict__ out_bx,
    uint64_t* __restrict__ out_ax,
    const uint64_t* __restrict__ mod,
    const uint64_t* __restrict__ barret_mu,
    const int64_t num_ciphers,
    const int64_t l,
    const int64_t N) {
  auto tid_x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  auto tid_y = blockIdx.y;

  uint64_t sum_bx = 0, sum_ax = 0;

  auto mod_val = mod[tid_y];
  auto barret_mu_val0 = barret_mu[tid_y * 2];
  auto barret_mu_val1 = barret_mu[tid_y * 2 + 1];

  GEN_MUL(0);
  GEN_MUL(1);
  GEN_MUL(2);
  GEN_MUL(3);
  GEN_MUL(4);
  GEN_MUL(5);
  GEN_MUL(6);
  GEN_MUL(7);
  GEN_MUL(8);

  GEN_SUM(0);
  GEN_SUM(1);
  GEN_SUM(2);
  GEN_SUM(3);
  GEN_SUM(4);
  GEN_SUM(5);
  GEN_SUM(6);
  GEN_SUM(7);
  GEN_SUM(8);

  out_bx[tid_y * N + tid_x] = sum_bx;
  out_ax[tid_y * N + tid_x] = sum_ax;
}

#undef GEN_MUL
#undef GEN_SUM
#undef GEN_PARAM

#define GEN_MUL(IDX)                                                          \
  auto ptx_val##IDX = ptx##IDX[tid_y * N + tid_x];                            \
  auto tmp_bx##IDX =                                                          \
      mul_mod(bx_val, ptx_val##IDX, mod_val, barret_mu_val0, barret_mu_val1); \
  auto tmp_ax##IDX =                                                          \
      mul_mod(ax_val, ptx_val##IDX, mod_val, barret_mu_val0, barret_mu_val1);

#define GEN_SUM(IDX)                              \
  sum_bx = add_mod(sum_bx, tmp_bx##IDX, mod_val); \
  sum_ax = add_mod(sum_ax, tmp_ax##IDX, mod_val)

__global__ void fusedBroadcastMACKernel16(
    const uint64_t* __restrict__ ptx0,
    const uint64_t* __restrict__ ptx1,
    const uint64_t* __restrict__ ptx2,
    const uint64_t* __restrict__ ptx3,
    const uint64_t* __restrict__ ptx4,
    const uint64_t* __restrict__ ptx5,
    const uint64_t* __restrict__ ptx6,
    const uint64_t* __restrict__ ptx7,
    const uint64_t* __restrict__ ptx8,
    const uint64_t* __restrict__ ptx9,
    const uint64_t* __restrict__ ptx10,
    const uint64_t* __restrict__ ptx11,
    const uint64_t* __restrict__ ptx12,
    const uint64_t* __restrict__ ptx13,
    const uint64_t* __restrict__ ptx14,
    const uint64_t* __restrict__ ptx15,
    const uint64_t* __restrict__ bx,
    const uint64_t* __restrict__ ax,
    uint64_t* __restrict__ out_bx,
    uint64_t* __restrict__ out_ax,
    const uint64_t* __restrict__ mod,
    const uint64_t* __restrict__ barret_mu,
    const int64_t num_ciphers,
    const int64_t l,
    const int64_t N) {
  auto tid_x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  auto tid_y = blockIdx.y;

  uint64_t sum_bx = 0, sum_ax = 0;

  auto mod_val = mod[tid_y];
  auto barret_mu_val0 = barret_mu[tid_y * 2];
  auto barret_mu_val1 = barret_mu[tid_y * 2 + 1];

  auto bx_val = bx[tid_y * N + tid_x];
  auto ax_val = ax[tid_y * N + tid_x];

  GEN_MUL(0);
  GEN_MUL(1);
  GEN_MUL(2);
  GEN_MUL(3);
  GEN_MUL(4);
  GEN_MUL(5);
  GEN_MUL(6);
  GEN_MUL(7);
  GEN_MUL(8);
  GEN_MUL(9);
  GEN_MUL(10);
  GEN_MUL(11);
  GEN_MUL(12);
  GEN_MUL(13);
  GEN_MUL(14);
  GEN_MUL(15);

  GEN_SUM(0);
  GEN_SUM(1);
  GEN_SUM(2);
  GEN_SUM(3);
  GEN_SUM(4);
  GEN_SUM(5);
  GEN_SUM(6);
  GEN_SUM(7);
  GEN_SUM(8);
  GEN_SUM(9);
  GEN_SUM(10);
  GEN_SUM(11);
  GEN_SUM(12);
  GEN_SUM(13);
  GEN_SUM(14);
  GEN_SUM(15);

  out_bx[tid_y * N + tid_x] += sum_bx;
  out_ax[tid_y * N + tid_x] += sum_ax;
}

} // namespace fhe

namespace at::native {

#define GEN_ARGS(IDX)                                           \
  bxs[IDX].data_ptr<uint64_t>(), axs[IDX].data_ptr<uint64_t>(), \
      ptxs[IDX].data_ptr<uint64_t>()

static void fused_pairwise_mac_template(
    Tensor& out_bx,
    Tensor& out_ax,
    TensorList bxs,
    TensorList axs,
    TensorList ptxs,
    const Tensor& param_primes,
    const Tensor& barret_mu,
    int64_t num_cipher,
    int64_t curr_limbs,
    int64_t N) {
  auto barret_mu_ptr = barret_mu.data_ptr<uint64_t>();
  auto param_primes_ptr = param_primes.data_ptr<uint64_t>();

  dim3 block(BLOCK_SIZE);
  dim3 grid(N / BLOCK_SIZE, curr_limbs);
  auto stream = at::cuda::getCurrentCUDAStream();

  fhe::fusedPairwiseMACKernel9<<<grid, block, 0, stream>>>(
      GEN_ARGS(0),
      GEN_ARGS(1),
      GEN_ARGS(2),
      GEN_ARGS(3),
      GEN_ARGS(4),
      GEN_ARGS(5),
      GEN_ARGS(6),
      GEN_ARGS(7),
      GEN_ARGS(8),
      out_bx.data_ptr<uint64_t>(),
      out_ax.data_ptr<uint64_t>(),
      param_primes_ptr,
      barret_mu_ptr,
      num_cipher,
      curr_limbs,
      N);
  cudaDeviceSynchronize();

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<Tensor> fused_pairwise_mac_cuda(
    TensorList bxs,
    TensorList axs,
    TensorList ptxs,
    const Tensor& param_primes,
    const Tensor& barret_mu,
    int64_t num_cipher,
    int64_t curr_limbs,
    int64_t N) {
  TORCH_CHECK(
      bxs.size() == 9,
      "fused_pairwise_mac only support 9, but got ",
      bxs.size());
  auto out_bx = at::empty(curr_limbs * N, bxs[0].options());
  auto out_ax = at::empty(curr_limbs * N, axs[0].options());

  fused_pairwise_mac_template(
      out_bx,
      out_ax,
      bxs,
      axs,
      ptxs,
      param_primes,
      barret_mu,
      num_cipher,
      curr_limbs,
      N);
  return {out_bx, out_ax};
}

static void fused_broadcast_mac_template(
    Tensor& out_bx,
    Tensor& out_ax,
    const Tensor& bx,
    const Tensor& ax,
    TensorList ptxs,
    const Tensor& param_primes,
    const Tensor& barret_mu,
    int64_t num_cipher,
    int64_t curr_limbs,
    int64_t N) {
  auto barret_mu_ptr = barret_mu.data_ptr<uint64_t>();
  auto param_primes_ptr = param_primes.data_ptr<uint64_t>();

  dim3 block(BLOCK_SIZE);
  dim3 grid(N / BLOCK_SIZE, curr_limbs);
  auto stream = at::cuda::getCurrentCUDAStream();

  for (int i = 0; i < num_cipher; i += 16) {
    fhe::fusedBroadcastMACKernel16<<<grid, block, 0, stream>>>(
        ptxs[i + 0].data_ptr<uint64_t>(),
        ptxs[i + 1].data_ptr<uint64_t>(),
        ptxs[i + 2].data_ptr<uint64_t>(),
        ptxs[i + 3].data_ptr<uint64_t>(),
        ptxs[i + 4].data_ptr<uint64_t>(),
        ptxs[i + 5].data_ptr<uint64_t>(),
        ptxs[i + 6].data_ptr<uint64_t>(),
        ptxs[i + 7].data_ptr<uint64_t>(),
        ptxs[i + 8].data_ptr<uint64_t>(),
        ptxs[i + 9].data_ptr<uint64_t>(),
        ptxs[i + 10].data_ptr<uint64_t>(),
        ptxs[i + 11].data_ptr<uint64_t>(),
        ptxs[i + 12].data_ptr<uint64_t>(),
        ptxs[i + 13].data_ptr<uint64_t>(),
        ptxs[i + 14].data_ptr<uint64_t>(),
        ptxs[i + 15].data_ptr<uint64_t>(),
        bx.data_ptr<uint64_t>(),
        ax.data_ptr<uint64_t>(),
        out_bx.data_ptr<uint64_t>(),
        out_ax.data_ptr<uint64_t>(),
        param_primes_ptr,
        barret_mu_ptr,
        num_cipher,
        curr_limbs,
        N);
  }
  cudaDeviceSynchronize();

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<Tensor> fused_broadcast_mac_cuda(
    const Tensor& bx,
    const Tensor& ax,
    TensorList ptxs,
    const Tensor& param_primes,
    const Tensor& barret_mu,
    int64_t num_cipher,
    int64_t curr_limbs,
    int64_t N) {
  TORCH_CHECK(
      ptxs.size() == 16 || ptxs.size() == 32 || ptxs.size() == 64,
      "fused_broadcast_mac only support 16/32/64, but got ",
      ptxs.size());

  auto out_bx = at::zeros(curr_limbs * N, bx[0].options());
  auto out_ax = at::zeros(curr_limbs * N, ax[0].options());

  fused_broadcast_mac_template(
      out_bx,
      out_ax,
      bx,
      ax,
      ptxs,
      param_primes,
      barret_mu,
      num_cipher,
      curr_limbs,
      N);
  return {out_bx, out_ax};
}

} // namespace at::native
