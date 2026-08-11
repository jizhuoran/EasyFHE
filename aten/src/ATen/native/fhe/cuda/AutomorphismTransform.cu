#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/empty.h>

#include <vector>

#include "ATen/native/fhe/cuda/device/Launch.cuh"
#include "ATen/native/fhe/cuda/device/Modular.cuh"

namespace fhe {
__global__ void finalize_fast_rotation_ext_kernel(
    uint64_t* __restrict__ out_bx,
    uint64_t* __restrict__ out_ax,
    const uint64_t* __restrict__ key_product_bx,
    const uint64_t* __restrict__ key_product_ax,
    const int64_t* __restrict__ product_indices,
    const uint64_t* __restrict__ c0,
    const uint64_t* __restrict__ c1,
    const int* __restrict__ precomp_maps,
    const uint64_t* __restrict__ primes,
    const uint64_t* __restrict__ p_mod_q,
    const uint64_t* __restrict__ barret_ks,
    const uint64_t* __restrict__ barret_ratios,
    const int64_t curr_limbs,
    const int64_t active_limbs,
    const int64_t N,
    const int64_t batch,
    const int64_t c_batch) {
  const int64_t j = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  const int64_t limb = blockIdx.y;
  const int64_t b = blockIdx.z;
  if (j >= N || limb >= active_limbs || b >= batch) {
    return;
  }

  const int64_t out_index = (b * active_limbs + limb) * N + j;
  const int64_t product_b = product_indices[b];
  if (product_b < 0) {
    if (limb < curr_limbs) {
      const int64_t coeff_index = limb * N + j;
      const int64_t c_offset = (c_batch == 1 ? 0 : b * curr_limbs * N);
      const auto prime = primes[limb];
      const auto p_mod_q_limb = p_mod_q[limb];
      out_bx[out_index] = barrett_reduction_128_64(
          mult_64_64_128(c0[c_offset + coeff_index], p_mod_q_limb),
          prime,
          barret_ratios[limb],
          barret_ks[limb]);
      out_ax[out_index] = barrett_reduction_128_64(
          mult_64_64_128(c1[c_offset + coeff_index], p_mod_q_limb),
          prime,
          barret_ratios[limb],
          barret_ks[limb]);
    } else {
      out_bx[out_index] = 0;
      out_ax[out_index] = 0;
    }
    return;
  }

  const int64_t src = precomp_maps[product_b * N + j];
  const int64_t key_index = (product_b * active_limbs + limb) * N + src;
  uint64_t bx = key_product_bx[key_index];
  if (limb < curr_limbs) {
    const int64_t coeff_index = limb * N + src;
    const int64_t c_offset = (c_batch == 1 ? 0 : b * curr_limbs * N);
    const auto prime = primes[limb];
    const auto scaled_c0 = barrett_reduction_128_64(
        mult_64_64_128(c0[c_offset + coeff_index], p_mod_q[limb]),
        prime,
        barret_ratios[limb],
        barret_ks[limb]);
    bx = add_mod(bx, scaled_c0, prime);
  }
  out_bx[out_index] = bx;
  out_ax[out_index] = key_product_ax[key_index];
}

__global__ void finalize_fast_rotation_q_kernel(
    uint64_t* __restrict__ out_bx,
    uint64_t* __restrict__ out_ax,
    const uint64_t* __restrict__ moddown_bx,
    const uint64_t* __restrict__ moddown_ax,
    const int64_t* __restrict__ product_indices,
    const uint64_t* __restrict__ c0,
    const uint64_t* __restrict__ c1,
    const int* __restrict__ precomp_maps,
    const uint64_t* __restrict__ primes,
    const int64_t curr_limbs,
    const int64_t N,
    const int64_t batch) {
  const int64_t j = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  const int64_t limb = blockIdx.y;
  const int64_t b = blockIdx.z;
  if (j >= N || limb >= curr_limbs || b >= batch) {
    return;
  }

  const int64_t out_index = (b * curr_limbs + limb) * N + j;
  const int64_t product_b = product_indices[b];
  if (product_b < 0) {
    out_bx[out_index] = c0[limb * N + j];
    out_ax[out_index] = c1[limb * N + j];
    return;
  }

  const int64_t src = precomp_maps[product_b * N + j];
  const int64_t product_index = (product_b * curr_limbs + limb) * N + src;
  out_bx[out_index] =
      add_mod(moddown_bx[product_index], c0[limb * N + src], primes[limb]);
  out_ax[out_index] = moddown_ax[product_index];
}

__global__ void double_hoist_giant_sum_ext_kernel(
    uint64_t* __restrict__ out_bx,
    uint64_t* __restrict__ out_ax,
    const uint64_t* __restrict__ base_bx,
    const uint64_t* __restrict__ base_ax,
    const uint64_t* __restrict__ key_product_bx,
    const uint64_t* __restrict__ key_product_ax,
    const uint64_t* __restrict__ c0,
    const int* __restrict__ precomp_maps,
    const uint64_t* __restrict__ primes,
    const uint64_t* __restrict__ p_mod_q,
    const uint64_t* __restrict__ barret_ks,
    const uint64_t* __restrict__ barret_ratios,
    const int64_t curr_limbs,
    const int64_t active_limbs,
    const int64_t N,
    const int64_t batch) {
  const int64_t j = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  const int64_t limb = blockIdx.y;
  if (j >= N || limb >= active_limbs) {
    return;
  }

  const int64_t out_index = limb * N + j;
  const auto prime = primes[limb];
  uint64_t bx = base_bx[out_index];
  uint64_t ax = base_ax[out_index];

  for (int64_t b = 0; b < batch; ++b) {
    const int64_t src = precomp_maps[b * N + j];
    const int64_t key_index = (b * active_limbs + limb) * N + src;
    uint64_t rotated_bx = key_product_bx[key_index];
    if (limb < curr_limbs) {
      const int64_t coeff_index = (b * curr_limbs + limb) * N + src;
      const auto scaled_c0 = barrett_reduction_128_64(
          mult_64_64_128(c0[coeff_index], p_mod_q[limb]),
          prime,
          barret_ratios[limb],
          barret_ks[limb]);
      rotated_bx = add_mod(rotated_bx, scaled_c0, prime);
    }
    bx = add_mod(bx, rotated_bx, prime);
    ax = add_mod(ax, key_product_ax[key_index], prime);
  }

  out_bx[out_index] = bx;
  out_ax[out_index] = ax;
}
} // namespace fhe

namespace at::native {
std::vector<Tensor> finalize_fast_rotation_ext_cuda(
    const Tensor& key_product_bx,
    const Tensor& key_product_ax,
    const Tensor& product_indices,
    const Tensor& c0,
    const Tensor& c1,
    const Tensor& precomp_maps,
    const Tensor& primes,
    const Tensor& p_mod_q,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    int64_t curr_limbs,
    int64_t active_limbs,
    int64_t N) {
  TORCH_INTERNAL_ASSERT(key_product_bx.dim() == 3);
  TORCH_INTERNAL_ASSERT(key_product_ax.dim() == 3);
  TORCH_INTERNAL_ASSERT(key_product_bx.sizes() == key_product_ax.sizes());
  TORCH_INTERNAL_ASSERT(key_product_bx.sizes()[1] == active_limbs);
  TORCH_INTERNAL_ASSERT(key_product_bx.sizes()[2] == N);
  TORCH_INTERNAL_ASSERT(product_indices.dim() == 1);
  TORCH_INTERNAL_ASSERT(product_indices.scalar_type() == at::kLong);
  const auto batch = product_indices.sizes()[0];
  const auto product_batch = key_product_bx.sizes()[0];
  TORCH_INTERNAL_ASSERT(c0.dim() == 3 && (c0.sizes()[0] == 1 || c0.sizes()[0] == batch));
  TORCH_INTERNAL_ASSERT(c1.dim() == 3 && (c1.sizes()[0] == 1 || c1.sizes()[0] == batch));
  TORCH_INTERNAL_ASSERT(c0.sizes()[1] >= curr_limbs);
  TORCH_INTERNAL_ASSERT(c0.sizes()[2] == N);
  TORCH_INTERNAL_ASSERT(c1.sizes()[1] >= curr_limbs);
  TORCH_INTERNAL_ASSERT(c1.sizes()[2] == N);
  TORCH_INTERNAL_ASSERT(precomp_maps.dim() == 2);
  TORCH_INTERNAL_ASSERT(precomp_maps.sizes()[1] == N);

  TORCH_INTERNAL_ASSERT(precomp_maps.sizes()[0] == product_batch);

  auto out_bx = at::empty({batch, active_limbs, N}, key_product_bx.options());
  auto out_ax = at::empty({batch, active_limbs, N}, key_product_ax.options());

  dim3 block(BLOCK_SIZE);
  dim3 grid(N / BLOCK_SIZE, active_limbs, batch);
  auto stream = at::cuda::getCurrentCUDAStream();

  fhe::finalize_fast_rotation_ext_kernel<<<grid, block, 0, stream>>>(
      out_bx.data_ptr<uint64_t>(),
      out_ax.data_ptr<uint64_t>(),
      key_product_bx.data_ptr<uint64_t>(),
      key_product_ax.data_ptr<uint64_t>(),
      product_indices.data_ptr<int64_t>(),
      c0.data_ptr<uint64_t>(),
      c1.data_ptr<uint64_t>(),
      precomp_maps.data_ptr<int>(),
      primes.data_ptr<uint64_t>(),
      p_mod_q.data_ptr<uint64_t>(),
      barret_k.data_ptr<uint64_t>(),
      barret_ratio.data_ptr<uint64_t>(),
      curr_limbs,
      active_limbs,
      N,
      batch,
      c0.sizes()[0]);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out_bx, out_ax};
}

std::vector<Tensor> finalize_fast_rotation_q_cuda(
    const Tensor& moddown_bx,
    const Tensor& moddown_ax,
    const Tensor& product_indices,
    const Tensor& c0,
    const Tensor& c1,
    const Tensor& precomp_maps,
    const Tensor& primes,
    int64_t curr_limbs,
    int64_t N) {
  TORCH_INTERNAL_ASSERT(moddown_bx.dim() == 3);
  TORCH_INTERNAL_ASSERT(moddown_ax.dim() == 3);
  TORCH_INTERNAL_ASSERT(moddown_bx.sizes() == moddown_ax.sizes());
  TORCH_INTERNAL_ASSERT(moddown_bx.sizes()[1] == curr_limbs);
  TORCH_INTERNAL_ASSERT(moddown_bx.sizes()[2] == N);
  TORCH_INTERNAL_ASSERT(product_indices.dim() == 1);
  TORCH_INTERNAL_ASSERT(product_indices.scalar_type() == at::kLong);
  TORCH_INTERNAL_ASSERT(c0.dim() == 3 && c0.sizes()[0] == 1);
  TORCH_INTERNAL_ASSERT(c1.dim() == 3 && c1.sizes()[0] == 1);
  TORCH_INTERNAL_ASSERT(c0.sizes()[1] >= curr_limbs);
  TORCH_INTERNAL_ASSERT(c0.sizes()[2] == N);
  TORCH_INTERNAL_ASSERT(c1.sizes()[1] >= curr_limbs);
  TORCH_INTERNAL_ASSERT(c1.sizes()[2] == N);
  TORCH_INTERNAL_ASSERT(precomp_maps.dim() == 2);
  TORCH_INTERNAL_ASSERT(precomp_maps.sizes()[1] == N);

  const auto batch = product_indices.sizes()[0];
  const auto product_batch = moddown_bx.sizes()[0];
  TORCH_INTERNAL_ASSERT(precomp_maps.sizes()[0] == product_batch);

  auto out_bx = at::empty({batch, curr_limbs, N}, moddown_bx.options());
  auto out_ax = at::empty({batch, curr_limbs, N}, moddown_ax.options());

  dim3 block(BLOCK_SIZE);
  dim3 grid(N / BLOCK_SIZE, curr_limbs, batch);
  auto stream = at::cuda::getCurrentCUDAStream();

  fhe::finalize_fast_rotation_q_kernel<<<grid, block, 0, stream>>>(
      out_bx.data_ptr<uint64_t>(),
      out_ax.data_ptr<uint64_t>(),
      moddown_bx.data_ptr<uint64_t>(),
      moddown_ax.data_ptr<uint64_t>(),
      product_indices.data_ptr<int64_t>(),
      c0.data_ptr<uint64_t>(),
      c1.data_ptr<uint64_t>(),
      precomp_maps.data_ptr<int>(),
      primes.data_ptr<uint64_t>(),
      curr_limbs,
      N,
      batch);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out_bx, out_ax};
}

std::vector<Tensor> double_hoist_giant_sum_ext_cuda(
    const Tensor& base_bx,
    const Tensor& base_ax,
    const Tensor& key_product_bx,
    const Tensor& key_product_ax,
    const Tensor& c0,
    const Tensor& precomp_maps,
    const Tensor& primes,
    const Tensor& p_mod_q,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    int64_t curr_limbs,
    int64_t active_limbs,
    int64_t N) {
  TORCH_INTERNAL_ASSERT(base_bx.dim() == 2);
  TORCH_INTERNAL_ASSERT(base_ax.dim() == 2);
  TORCH_INTERNAL_ASSERT(base_bx.sizes() == base_ax.sizes());
  TORCH_INTERNAL_ASSERT(base_bx.sizes()[0] == active_limbs);
  TORCH_INTERNAL_ASSERT(base_bx.sizes()[1] == N);
  TORCH_INTERNAL_ASSERT(key_product_bx.dim() == 3);
  TORCH_INTERNAL_ASSERT(key_product_ax.dim() == 3);
  TORCH_INTERNAL_ASSERT(key_product_bx.sizes() == key_product_ax.sizes());
  TORCH_INTERNAL_ASSERT(key_product_bx.sizes()[1] == active_limbs);
  TORCH_INTERNAL_ASSERT(key_product_bx.sizes()[2] == N);
  TORCH_INTERNAL_ASSERT(c0.dim() == 3);
  TORCH_INTERNAL_ASSERT(c0.sizes()[1] >= curr_limbs);
  TORCH_INTERNAL_ASSERT(c0.sizes()[2] == N);
  TORCH_INTERNAL_ASSERT(precomp_maps.dim() == 2);
  TORCH_INTERNAL_ASSERT(precomp_maps.sizes()[1] == N);

  const auto batch = key_product_bx.sizes()[0];
  TORCH_INTERNAL_ASSERT(batch == c0.sizes()[0]);
  TORCH_INTERNAL_ASSERT(batch == precomp_maps.sizes()[0]);

  auto out_bx = at::empty({active_limbs, N}, base_bx.options());
  auto out_ax = at::empty({active_limbs, N}, base_ax.options());

  dim3 block(BLOCK_SIZE);
  dim3 grid(N / BLOCK_SIZE, active_limbs);
  auto stream = at::cuda::getCurrentCUDAStream();

  fhe::double_hoist_giant_sum_ext_kernel<<<grid, block, 0, stream>>>(
      out_bx.data_ptr<uint64_t>(),
      out_ax.data_ptr<uint64_t>(),
      base_bx.data_ptr<uint64_t>(),
      base_ax.data_ptr<uint64_t>(),
      key_product_bx.data_ptr<uint64_t>(),
      key_product_ax.data_ptr<uint64_t>(),
      c0.data_ptr<uint64_t>(),
      precomp_maps.data_ptr<int>(),
      primes.data_ptr<uint64_t>(),
      p_mod_q.data_ptr<uint64_t>(),
      barret_k.data_ptr<uint64_t>(),
      barret_ratio.data_ptr<uint64_t>(),
      curr_limbs,
      active_limbs,
      N,
      batch);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out_bx, out_ax};
}

} // namespace at::native
