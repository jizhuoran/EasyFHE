#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>

#include <vector>

#include "ATen/native/fhe/cuda/Utils.cuh"

namespace fhe {
template <size_t NUM_CV>
__global__ void automorphism_transform_kernel(
    uint64_t* __restrict__ out,
    const uint64_t* __restrict__ in,
    const size_t N,
    const size_t LN,
    const size_t BLN,
    const int32_t* __restrict__ precomp_vec) {
  const size_t coeff = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  const size_t limb = blockIdx.y;
  const size_t batch = blockIdx.z;
  if (coeff >= N) {
    return;
  }

  const size_t src = static_cast<size_t>(precomp_vec[coeff]);
  const size_t batch_offset = batch * LN;
  const size_t limb_offset = limb * N;
#pragma unroll
  for (int cv = 0; cv < NUM_CV; ++cv) {
    const size_t cv_offset = cv * BLN + batch_offset + limb_offset;
    out[cv_offset + coeff] = in[cv_offset + src];
  }
}
} // namespace fhe

namespace fhe {
__global__ void fast_rotate_ext_batch_finalize_kernel(
    uint64_t* __restrict__ out_bx,
    uint64_t* __restrict__ out_ax,
    const uint64_t* __restrict__ key_products,
    const uint64_t* __restrict__ pc0,
    const uint64_t* __restrict__ pc1,
    const int* __restrict__ precomp_maps,
    const int64_t* __restrict__ offsets,
    const uint64_t* __restrict__ primes,
    const int64_t curr_limbs,
    const int64_t active_limbs,
    const int64_t N,
    const int64_t batch) {
  const int64_t j = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  const int64_t limb = blockIdx.y;
  const int64_t b = blockIdx.z;
  if (j >= N || limb >= active_limbs || b >= batch) {
    return;
  }

  const int64_t out_index = (b * active_limbs + limb) * N + j;
  if (offsets[b] == 0) {
    out_bx[out_index] = pc0[limb * N + j];
    out_ax[out_index] = pc1[limb * N + j];
    return;
  }

  const int64_t src = precomp_maps[b * N + j];
  const int64_t key_stride = batch * active_limbs * N;
  const int64_t key_index = (b * active_limbs + limb) * N + src;
  uint64_t bx = key_products[key_index];
  if (limb < curr_limbs) {
    bx = add_mod(bx, pc0[limb * N + src], primes[limb]);
  }
  out_bx[out_index] = bx;
  out_ax[out_index] = key_products[key_stride + key_index];
}

__global__ void fast_rotate_ext_batch_finalize_compact_kernel(
    uint64_t* __restrict__ out_bx,
    uint64_t* __restrict__ out_ax,
    const uint64_t* __restrict__ key_products,
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
    const int64_t product_batch) {
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
      const auto prime = primes[limb];
      const auto barret_ratio = barret_ratios[limb];
      const auto barret_k = barret_ks[limb];
      const auto p_mod_q_limb = p_mod_q[limb];
      out_bx[out_index] = barret_reduction_128_64(
          mult_64_64_128(c0[coeff_index], p_mod_q_limb),
          prime,
          barret_ratio,
          barret_k);
      out_ax[out_index] = barret_reduction_128_64(
          mult_64_64_128(c1[coeff_index], p_mod_q_limb),
          prime,
          barret_ratio,
          barret_k);
    } else {
      out_bx[out_index] = 0;
      out_ax[out_index] = 0;
    }
    return;
  }

  const int64_t src = precomp_maps[b * N + j];
  const int64_t key_stride = product_batch * active_limbs * N;
  const int64_t key_index = (product_b * active_limbs + limb) * N + src;
  uint64_t bx = key_products[key_index];
  if (limb < curr_limbs) {
    const int64_t coeff_index = limb * N + src;
    const auto prime = primes[limb];
    const auto scaled_c0 = barret_reduction_128_64(
        mult_64_64_128(c0[coeff_index], p_mod_q[limb]),
        prime,
        barret_ratios[limb],
        barret_ks[limb]);
    bx = add_mod(bx, scaled_c0, prime);
  }
  out_bx[out_index] = bx;
  out_ax[out_index] = key_products[key_stride + key_index];
}

__global__ void fast_rotate_ext_batch_finalize_pair_kernel(
    uint64_t* __restrict__ out_bx,
    uint64_t* __restrict__ out_ax,
    const uint64_t* __restrict__ key_product_bx,
    const uint64_t* __restrict__ key_product_ax,
    const uint64_t* __restrict__ pc0,
    const uint64_t* __restrict__ pc1,
    const int* __restrict__ precomp_maps,
    const int64_t* __restrict__ offsets,
    const uint64_t* __restrict__ primes,
    const int64_t curr_limbs,
    const int64_t active_limbs,
    const int64_t N,
    const int64_t batch) {
  const int64_t j = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  const int64_t limb = blockIdx.y;
  const int64_t b = blockIdx.z;
  if (j >= N || limb >= active_limbs || b >= batch) {
    return;
  }

  const int64_t out_index = (b * active_limbs + limb) * N + j;
  if (offsets[b] == 0) {
    out_bx[out_index] = pc0[limb * N + j];
    out_ax[out_index] = pc1[limb * N + j];
    return;
  }

  const int64_t src = precomp_maps[b * N + j];
  const int64_t key_index = (b * active_limbs + limb) * N + src;
  uint64_t bx = key_product_bx[key_index];
  if (limb < curr_limbs) {
    bx = add_mod(bx, pc0[limb * N + src], primes[limb]);
  }
  out_bx[out_index] = bx;
  out_ax[out_index] = key_product_ax[key_index];
}

__global__ void fast_rotate_batch_finalize_kernel(
    uint64_t* __restrict__ out_bx,
    uint64_t* __restrict__ out_ax,
    const uint64_t* __restrict__ moddown_products,
    const uint64_t* __restrict__ c0,
    const uint64_t* __restrict__ c1,
    const int* __restrict__ precomp_maps,
    const int64_t* __restrict__ offsets,
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
  if (offsets[b] == 0) {
    out_bx[out_index] = c0[limb * N + j];
    out_ax[out_index] = c1[limb * N + j];
    return;
  }

  const int64_t src = precomp_maps[b * N + j];
  const int64_t product_stride = batch * curr_limbs * N;
  const int64_t product_index = (b * curr_limbs + limb) * N + src;
  out_bx[out_index] =
      add_mod(moddown_products[product_index], c0[limb * N + src], primes[limb]);
  out_ax[out_index] = moddown_products[product_stride + product_index];
}

__global__ void fast_rotate_batch_finalize_compact_kernel(
    uint64_t* __restrict__ out_bx,
    uint64_t* __restrict__ out_ax,
    const uint64_t* __restrict__ moddown_products,
    const int64_t* __restrict__ product_indices,
    const uint64_t* __restrict__ c0,
    const uint64_t* __restrict__ c1,
    const int* __restrict__ precomp_maps,
    const uint64_t* __restrict__ primes,
    const int64_t curr_limbs,
    const int64_t N,
    const int64_t batch,
    const int64_t product_batch) {
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

  const int64_t src = precomp_maps[b * N + j];
  const int64_t product_stride = product_batch * curr_limbs * N;
  const int64_t product_index = (product_b * curr_limbs + limb) * N + src;
  out_bx[out_index] =
      add_mod(moddown_products[product_index], c0[limb * N + src], primes[limb]);
  out_ax[out_index] = moddown_products[product_stride + product_index];
}
} // namespace fhe

namespace at::native {
static void automorphism_transform_template(
    Tensor& out,
    const Tensor& in,
    int64_t l,
    int64_t N,
    const Tensor& precomp_vec) {
  auto* out_ptr = out.data_ptr<uint64_t>();
  const auto* in_ptr = in.data_ptr<uint64_t>();
  const auto* precomp_vec_ptr = precomp_vec.data_ptr<int32_t>();

  const dim3 block(BLOCK_SIZE);
  const dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, l, out.sizes()[1]);
  auto stream = at::cuda::getCurrentCUDAStream();

  const auto LN = in.sizes()[2] * N;
  const auto BLN = in.sizes()[1] * LN;

  if (out.sizes()[0] == 1) {
    fhe::automorphism_transform_kernel<1><<<grid, block, 0, stream>>>(
        out_ptr, in_ptr, N, LN, BLN, precomp_vec_ptr);
  } else if (out.sizes()[0] == 2) {
    fhe::automorphism_transform_kernel<2><<<grid, block, 0, stream>>>(
        out_ptr, in_ptr, N, LN, BLN, precomp_vec_ptr);
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
  TORCH_INTERNAL_ASSERT(l <= from.sizes()[2]);
  TORCH_INTERNAL_ASSERT(N == from.sizes()[3]);
  TORCH_INTERNAL_ASSERT(precomp_vec.numel() == N);

  Tensor out = at::empty_like(from);
  automorphism_transform_template(out, from, l, N, precomp_vec);
  return out;
}

std::vector<Tensor> fast_rotate_ext_batch_finalize_cuda(
    const Tensor& key_products,
    const Tensor& pc0,
    const Tensor& pc1,
    const Tensor& precomp_maps,
    const Tensor& offsets,
    const Tensor& primes,
    int64_t curr_limbs,
    int64_t active_limbs,
    int64_t N) {
  TORCH_INTERNAL_ASSERT(key_products.dim() == 4);
  TORCH_INTERNAL_ASSERT(key_products.sizes()[0] == 2);
  TORCH_INTERNAL_ASSERT(key_products.sizes()[2] == active_limbs);
  TORCH_INTERNAL_ASSERT(key_products.sizes()[3] == N);
  TORCH_INTERNAL_ASSERT(pc0.sizes()[0] == active_limbs);
  TORCH_INTERNAL_ASSERT(pc0.sizes()[1] == N);
  TORCH_INTERNAL_ASSERT(pc1.sizes()[0] == active_limbs);
  TORCH_INTERNAL_ASSERT(pc1.sizes()[1] == N);
  TORCH_INTERNAL_ASSERT(precomp_maps.dim() == 2);
  TORCH_INTERNAL_ASSERT(precomp_maps.sizes()[1] == N);
  TORCH_INTERNAL_ASSERT(offsets.dim() == 1);

  const auto batch = key_products.sizes()[1];
  TORCH_INTERNAL_ASSERT(precomp_maps.sizes()[0] == batch);
  TORCH_INTERNAL_ASSERT(offsets.sizes()[0] == batch);

  auto out_bx = at::empty({batch, active_limbs, N}, key_products.options());
  auto out_ax = at::empty({batch, active_limbs, N}, key_products.options());

  dim3 block(BLOCK_SIZE);
  dim3 grid(N / BLOCK_SIZE, active_limbs, batch);
  auto stream = at::cuda::getCurrentCUDAStream();

  fhe::fast_rotate_ext_batch_finalize_kernel<<<grid, block, 0, stream>>>(
      out_bx.data_ptr<uint64_t>(),
      out_ax.data_ptr<uint64_t>(),
      key_products.data_ptr<uint64_t>(),
      pc0.data_ptr<uint64_t>(),
      pc1.data_ptr<uint64_t>(),
      precomp_maps.data_ptr<int>(),
      offsets.data_ptr<int64_t>(),
      primes.data_ptr<uint64_t>(),
      curr_limbs,
      active_limbs,
      N,
      batch);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out_bx, out_ax};
}

std::vector<Tensor> fast_rotate_ext_batch_finalize_compact_cuda(
    const Tensor& key_products,
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
  TORCH_INTERNAL_ASSERT(key_products.dim() == 4);
  TORCH_INTERNAL_ASSERT(key_products.sizes()[0] == 2);
  TORCH_INTERNAL_ASSERT(key_products.sizes()[2] == active_limbs);
  TORCH_INTERNAL_ASSERT(key_products.sizes()[3] == N);
  TORCH_INTERNAL_ASSERT(product_indices.dim() == 1);
  TORCH_INTERNAL_ASSERT(product_indices.scalar_type() == at::kLong);
  TORCH_INTERNAL_ASSERT(c0.sizes()[0] == curr_limbs);
  TORCH_INTERNAL_ASSERT(c0.sizes()[1] == N);
  TORCH_INTERNAL_ASSERT(c1.sizes()[0] == curr_limbs);
  TORCH_INTERNAL_ASSERT(c1.sizes()[1] == N);
  TORCH_INTERNAL_ASSERT(precomp_maps.dim() == 2);
  TORCH_INTERNAL_ASSERT(precomp_maps.sizes()[1] == N);
  TORCH_INTERNAL_ASSERT(p_mod_q.numel() >= curr_limbs);
  TORCH_INTERNAL_ASSERT(barret_ratio.numel() >= curr_limbs);
  TORCH_INTERNAL_ASSERT(barret_k.numel() >= curr_limbs);

  const auto batch = product_indices.sizes()[0];
  const auto product_batch = key_products.sizes()[1];
  TORCH_INTERNAL_ASSERT(precomp_maps.sizes()[0] == batch);

  auto out_bx = at::empty({batch, active_limbs, N}, key_products.options());
  auto out_ax = at::empty({batch, active_limbs, N}, key_products.options());

  dim3 block(BLOCK_SIZE);
  dim3 grid(N / BLOCK_SIZE, active_limbs, batch);
  auto stream = at::cuda::getCurrentCUDAStream();

  fhe::fast_rotate_ext_batch_finalize_compact_kernel<<<grid, block, 0, stream>>>(
      out_bx.data_ptr<uint64_t>(),
      out_ax.data_ptr<uint64_t>(),
      key_products.data_ptr<uint64_t>(),
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
      product_batch);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out_bx, out_ax};
}

std::vector<Tensor> fast_rotate_ext_batch_finalize_pair_cuda(
    const Tensor& key_product_bx,
    const Tensor& key_product_ax,
    const Tensor& pc0,
    const Tensor& pc1,
    const Tensor& precomp_maps,
    const Tensor& offsets,
    const Tensor& primes,
    int64_t curr_limbs,
    int64_t active_limbs,
    int64_t N) {
  TORCH_INTERNAL_ASSERT(key_product_bx.dim() == 3);
  TORCH_INTERNAL_ASSERT(key_product_ax.dim() == 3);
  TORCH_INTERNAL_ASSERT(key_product_bx.sizes() == key_product_ax.sizes());
  TORCH_INTERNAL_ASSERT(key_product_bx.sizes()[1] == active_limbs);
  TORCH_INTERNAL_ASSERT(key_product_bx.sizes()[2] == N);
  TORCH_INTERNAL_ASSERT(pc0.sizes()[0] == active_limbs);
  TORCH_INTERNAL_ASSERT(pc0.sizes()[1] == N);
  TORCH_INTERNAL_ASSERT(pc1.sizes()[0] == active_limbs);
  TORCH_INTERNAL_ASSERT(pc1.sizes()[1] == N);
  TORCH_INTERNAL_ASSERT(precomp_maps.dim() == 2);
  TORCH_INTERNAL_ASSERT(precomp_maps.sizes()[1] == N);
  TORCH_INTERNAL_ASSERT(offsets.dim() == 1);

  const auto batch = key_product_bx.sizes()[0];
  TORCH_INTERNAL_ASSERT(precomp_maps.sizes()[0] == batch);
  TORCH_INTERNAL_ASSERT(offsets.sizes()[0] == batch);

  auto out_bx = at::empty({batch, active_limbs, N}, key_product_bx.options());
  auto out_ax = at::empty({batch, active_limbs, N}, key_product_ax.options());

  dim3 block(BLOCK_SIZE);
  dim3 grid(N / BLOCK_SIZE, active_limbs, batch);
  auto stream = at::cuda::getCurrentCUDAStream();

  fhe::fast_rotate_ext_batch_finalize_pair_kernel<<<grid, block, 0, stream>>>(
      out_bx.data_ptr<uint64_t>(),
      out_ax.data_ptr<uint64_t>(),
      key_product_bx.data_ptr<uint64_t>(),
      key_product_ax.data_ptr<uint64_t>(),
      pc0.data_ptr<uint64_t>(),
      pc1.data_ptr<uint64_t>(),
      precomp_maps.data_ptr<int>(),
      offsets.data_ptr<int64_t>(),
      primes.data_ptr<uint64_t>(),
      curr_limbs,
      active_limbs,
      N,
      batch);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out_bx, out_ax};
}

std::vector<Tensor> fast_rotate_batch_finalize_cuda(
    const Tensor& moddown_products,
    const Tensor& c0,
    const Tensor& c1,
    const Tensor& precomp_maps,
    const Tensor& offsets,
    const Tensor& primes,
    int64_t curr_limbs,
    int64_t N) {
  TORCH_INTERNAL_ASSERT(moddown_products.dim() == 4);
  TORCH_INTERNAL_ASSERT(moddown_products.sizes()[0] == 2);
  TORCH_INTERNAL_ASSERT(moddown_products.sizes()[2] == curr_limbs);
  TORCH_INTERNAL_ASSERT(moddown_products.sizes()[3] == N);
  TORCH_INTERNAL_ASSERT(c0.sizes()[0] == curr_limbs);
  TORCH_INTERNAL_ASSERT(c0.sizes()[1] == N);
  TORCH_INTERNAL_ASSERT(c1.sizes()[0] == curr_limbs);
  TORCH_INTERNAL_ASSERT(c1.sizes()[1] == N);
  TORCH_INTERNAL_ASSERT(precomp_maps.dim() == 2);
  TORCH_INTERNAL_ASSERT(precomp_maps.sizes()[1] == N);
  TORCH_INTERNAL_ASSERT(offsets.dim() == 1);

  const auto batch = moddown_products.sizes()[1];
  TORCH_INTERNAL_ASSERT(precomp_maps.sizes()[0] == batch);
  TORCH_INTERNAL_ASSERT(offsets.sizes()[0] == batch);

  auto out_bx = at::empty({batch, curr_limbs, N}, moddown_products.options());
  auto out_ax = at::empty({batch, curr_limbs, N}, moddown_products.options());

  dim3 block(BLOCK_SIZE);
  dim3 grid(N / BLOCK_SIZE, curr_limbs, batch);
  auto stream = at::cuda::getCurrentCUDAStream();

  fhe::fast_rotate_batch_finalize_kernel<<<grid, block, 0, stream>>>(
      out_bx.data_ptr<uint64_t>(),
      out_ax.data_ptr<uint64_t>(),
      moddown_products.data_ptr<uint64_t>(),
      c0.data_ptr<uint64_t>(),
      c1.data_ptr<uint64_t>(),
      precomp_maps.data_ptr<int>(),
      offsets.data_ptr<int64_t>(),
      primes.data_ptr<uint64_t>(),
      curr_limbs,
      N,
      batch);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out_bx, out_ax};
}

std::vector<Tensor> fast_rotate_batch_finalize_compact_cuda(
    const Tensor& moddown_products,
    const Tensor& product_indices,
    const Tensor& c0,
    const Tensor& c1,
    const Tensor& precomp_maps,
    const Tensor& primes,
    int64_t curr_limbs,
    int64_t N) {
  TORCH_INTERNAL_ASSERT(moddown_products.dim() == 4);
  TORCH_INTERNAL_ASSERT(moddown_products.sizes()[0] == 2);
  TORCH_INTERNAL_ASSERT(moddown_products.sizes()[2] == curr_limbs);
  TORCH_INTERNAL_ASSERT(moddown_products.sizes()[3] == N);
  TORCH_INTERNAL_ASSERT(product_indices.dim() == 1);
  TORCH_INTERNAL_ASSERT(product_indices.scalar_type() == at::kLong);
  TORCH_INTERNAL_ASSERT(c0.sizes()[0] == curr_limbs);
  TORCH_INTERNAL_ASSERT(c0.sizes()[1] == N);
  TORCH_INTERNAL_ASSERT(c1.sizes()[0] == curr_limbs);
  TORCH_INTERNAL_ASSERT(c1.sizes()[1] == N);
  TORCH_INTERNAL_ASSERT(precomp_maps.dim() == 2);
  TORCH_INTERNAL_ASSERT(precomp_maps.sizes()[1] == N);

  const auto batch = product_indices.sizes()[0];
  const auto product_batch = moddown_products.sizes()[1];
  TORCH_INTERNAL_ASSERT(precomp_maps.sizes()[0] == batch);

  auto out_bx = at::empty({batch, curr_limbs, N}, moddown_products.options());
  auto out_ax = at::empty({batch, curr_limbs, N}, moddown_products.options());

  dim3 block(BLOCK_SIZE);
  dim3 grid(N / BLOCK_SIZE, curr_limbs, batch);
  auto stream = at::cuda::getCurrentCUDAStream();

  fhe::fast_rotate_batch_finalize_compact_kernel<<<grid, block, 0, stream>>>(
      out_bx.data_ptr<uint64_t>(),
      out_ax.data_ptr<uint64_t>(),
      moddown_products.data_ptr<uint64_t>(),
      product_indices.data_ptr<int64_t>(),
      c0.data_ptr<uint64_t>(),
      c1.data_ptr<uint64_t>(),
      precomp_maps.data_ptr<int>(),
      primes.data_ptr<uint64_t>(),
      curr_limbs,
      N,
      batch,
      product_batch);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out_bx, out_ax};
}

} // namespace at::native
