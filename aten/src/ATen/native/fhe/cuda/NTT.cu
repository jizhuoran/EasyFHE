#include <ATen/Dispatch_v2.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/thread_constants.h>
#include <cstdint>

#include "ATen/native/fhe/cuda/arithmetic.cuh"

namespace fhe {
__device__ void CooleyTukeyUnit(
    uint64_t& U,
    uint64_t& V,
    uint64_t& root,
    uint64_t mod,
    uint64_t barret_mu0,
    uint64_t barret_mu1) {
  uint64_t u_ = U;
  uint64_t v_ = mul_mod(V, root, mod, barret_mu0, barret_mu1);

  U = add_mod(u_, v_, mod);
  V = sub_mod(u_, v_, mod);
}

__device__ void GentlemanSandeUnit(
    uint64_t& U,
    uint64_t& V,
    uint64_t& root,
    uint64_t mod,
    uint64_t barret_mu0,
    uint64_t barret_mu1) {
  uint64_t u_ = U;
  uint64_t v_ = V;

  U = add_mod(u_, v_, mod);
  v_ = sub_mod(u_, v_, mod);
  V = mul_mod(v_, root, mod, barret_mu0, barret_mu1);
}

template <
    bool not_last_kernel,
    int shared_index,
    int logm,
    int outer_iteration_count,
    int N_power>
__global__ void ForwardCore(
    uint64_t* polynomial_in,
    uint64_t* polynomial_out,
    uint64_t* root_of_unity_table,
    uint64_t mod,
    uint64_t* barret_mu) {
  const int idx_x = threadIdx.x;
  const int idx_y = threadIdx.y;
  const int block_x = blockIdx.x;
  const int block_y = blockIdx.y;
  const int block_z = blockIdx.z;

  uint64_t barret_mu0 = barret_mu[0];
  uint64_t barret_mu1 = barret_mu[1];

  extern __shared__ uint64_t shared_memory[];

  int t_2 = N_power - logm - 1;
  size_t offset = 1 << (N_power - logm - 1);
  int t_ = shared_index;
  size_t m = (size_t)1 << logm;

  size_t global_addresss = idx_x +
      (size_t)(idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
      (size_t)(blockDim.x * block_x) + (size_t)(2 * block_y * offset) +
      (size_t)(block_z << N_power);

  size_t omega_addresss = idx_x +
      (size_t)(idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
      (size_t)(blockDim.x * block_x) + (size_t)(block_y * offset);

  size_t shared_addresss = (idx_x + (idx_y * blockDim.x));

  // Load data from global & store to shared
  shared_memory[shared_addresss] = polynomial_in[global_addresss];
  shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
      polynomial_in[global_addresss + offset];

  int t = 1 << t_;
  int in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
  size_t current_root_index;
  if (not_last_kernel) {
#pragma unroll
    for (int lp = 0; lp < outer_iteration_count; lp++) {
      __syncthreads();
      current_root_index = m + (omega_addresss >> t_2);
      CooleyTukeyUnit(
          shared_memory[in_shared_address],
          shared_memory[in_shared_address + t],
          root_of_unity_table[current_root_index],
          mod,
          barret_mu0,
          barret_mu1);

      t = t >> 1;
      t_2 -= 1;
      t_ -= 1;
      m <<= 1;

      in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
      //__syncthreads();
    }
    __syncthreads();
  } else {
#pragma unroll
    for (int lp = 0; lp < (shared_index - 5); lp++) // 4 for 512 thread
    {
      __syncthreads();
      current_root_index = m + (omega_addresss >> t_2);

      CooleyTukeyUnit(
          shared_memory[in_shared_address],
          shared_memory[in_shared_address + t],
          root_of_unity_table[current_root_index],
          mod,
          barret_mu0,
          barret_mu1);

      t = t >> 1;
      t_2 -= 1;
      t_ -= 1;
      m <<= 1;

      in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
      //__syncthreads();
    }
    __syncthreads();

#pragma unroll
    for (int lp = 0; lp < 6; lp++) {
      current_root_index = m + (omega_addresss >> t_2);
      CooleyTukeyUnit(
          shared_memory[in_shared_address],
          shared_memory[in_shared_address + t],
          root_of_unity_table[current_root_index],
          mod,
          barret_mu0,
          barret_mu1);

      t = t >> 1;
      t_2 -= 1;
      t_ -= 1;
      m <<= 1;

      in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
    }
    __syncthreads();
  }

  polynomial_out[global_addresss] = shared_memory[shared_addresss];
  polynomial_out[global_addresss + offset] =
      shared_memory[shared_addresss + (blockDim.x * blockDim.y)];
}

template <
    bool last_kernel,
    int shared_index,
    int logm,
    int k,
    int outer_iteration_count,
    int N_power>
__global__ void InverseCore(
    uint64_t* polynomial_in,
    uint64_t* polynomial_out,
    uint64_t* inverse_root_of_unity_table,
    uint64_t mod,
    uint64_t* barret_mu,
    uint64_t n_inverse) {
  const int idx_x = threadIdx.x;
  const int idx_y = threadIdx.y;
  const int block_x = blockIdx.x;
  const int block_y = blockIdx.y;
  const int block_z = blockIdx.z;

  uint64_t barret_mu0 = barret_mu[0];
  uint64_t barret_mu1 = barret_mu[1];

  extern __shared__ uint64_t shared_memory[];

  int t_2 = N_power - logm - 1;
  size_t offset = 1 << (N_power - k - 1);
  // int t_ = 9 - outer_iteration_count;
  int t_ = (shared_index + 1) - outer_iteration_count;
  int loops = outer_iteration_count;
  size_t m = (size_t)1 << logm;

  size_t global_addresss = idx_x +
      (size_t)(idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
      (size_t)(blockDim.x * block_x) + (size_t)(2 * block_y * offset) +
      (size_t)(block_z << N_power);

  size_t omega_addresss = idx_x +
      (size_t)(idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
      (size_t)(blockDim.x * block_x) + (size_t)(block_y * offset);
  size_t shared_addresss = (idx_x + (idx_y * blockDim.x));

  shared_memory[shared_addresss] = polynomial_in[global_addresss];
  shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
      polynomial_in[global_addresss + offset];

  int t = 1 << t_;
  int in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
  size_t current_root_index;
#pragma unroll
  for (int lp = 0; lp < loops; lp++) {
    __syncthreads();
    current_root_index = m + (omega_addresss >> t_2);

    GentlemanSandeUnit(
        shared_memory[in_shared_address],
        shared_memory[in_shared_address + t],
        inverse_root_of_unity_table[current_root_index],
        mod,
        barret_mu0,
        barret_mu1);

    t = t << 1;
    t_2 += 1;
    t_ += 1;
    m >>= 1;

    in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
  }
  __syncthreads();

  if (last_kernel) {
    polynomial_out[global_addresss] = mul_mod(
        shared_memory[shared_addresss], n_inverse, mod, barret_mu0, barret_mu1);
    polynomial_out[global_addresss + offset] = mul_mod(
        shared_memory[shared_addresss + (blockDim.x * blockDim.y)],
        n_inverse,
        mod,
        barret_mu0,
        barret_mu1);
  } else {
    polynomial_out[global_addresss] = shared_memory[shared_addresss];
    polynomial_out[global_addresss + offset] =
        shared_memory[shared_addresss + (blockDim.x * blockDim.y)];
  }
}

__host__ void NTT(
    size_t n_power,
    uint64_t* in,
    uint64_t* out,
    uint64_t* root_of_unity_table,
    uint64_t mod,
    uint64_t* barret_mu,
    int batch_size = 1) {
  switch (n_power) {
    case 15:
      ForwardCore<true, 8, 0, 6, 15>
          <<<dim3(64, 1, batch_size), dim3(8, 32), 512 * sizeof(uint64_t)>>>(
              in, out, root_of_unity_table, mod, barret_mu);
      C10_CUDA_KERNEL_LAUNCH_CHECK();

      ForwardCore<false, 8, 6, 9, 15>
          <<<dim3(1, 64, batch_size), dim3(256, 1), 512 * sizeof(uint64_t)>>>(
              out, out, root_of_unity_table, mod, barret_mu);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
    case 16:
      ForwardCore<true, 8, 0, 7, 16>
          <<<dim3(128, 1, batch_size), dim3(4, 64), 512 * sizeof(uint64_t)>>>(
              in, out, root_of_unity_table, mod, barret_mu);
      C10_CUDA_KERNEL_LAUNCH_CHECK();

      ForwardCore<false, 8, 7, 9, 16>
          <<<dim3(1, 128, batch_size), dim3(256, 1), 512 * sizeof(uint64_t)>>>(
              out, out, root_of_unity_table, mod, barret_mu);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
    case 17:
      ForwardCore<true, 8, 0, 4, 17>
          <<<dim3(256, 1, batch_size), dim3(32, 8), 512 * sizeof(uint64_t)>>>(
              in, out, root_of_unity_table, mod, barret_mu);
      C10_CUDA_KERNEL_LAUNCH_CHECK();

      ForwardCore<true, 8, 4, 4, 17>
          <<<dim3(16, 16, batch_size), dim3(32, 8), 512 * sizeof(uint64_t)>>>(
              in, out, root_of_unity_table, mod, barret_mu);
      C10_CUDA_KERNEL_LAUNCH_CHECK();

      ForwardCore<false, 8, 8, 9, 17>
          <<<dim3(1, 256, batch_size), dim3(256, 1), 512 * sizeof(uint64_t)>>>(
              in, out, root_of_unity_table, mod, barret_mu);
      C10_CUDA_KERNEL_LAUNCH_CHECK();

      break;
    default:
      TORCH_INTERNAL_ASSERT(false, "Invalid n_power");
  }
}

__host__ void InverseNTT(
    size_t n_power,
    uint64_t* in,
    uint64_t* out,
    uint64_t* root_of_unity_table,
    uint64_t mod,
    uint64_t* barret_mu,
    uint64_t n_inverse,
    int batch_size = 1) {
  switch (n_power) {
    case 15:
      InverseCore<false, 8, 14, 6, 9, 15>
          <<<dim3(1, 64, batch_size), dim3(256, 1), 512 * sizeof(uint64_t)>>>(
              in, out, root_of_unity_table, mod, barret_mu, n_inverse);
      C10_CUDA_KERNEL_LAUNCH_CHECK();

      InverseCore<true, 8, 5, 0, 6, 15>
          <<<dim3(64, 1, batch_size), dim3(8, 32), 512 * sizeof(uint64_t)>>>(
              out, out, root_of_unity_table, mod, barret_mu, n_inverse);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
    case 16:
      InverseCore<false, 8, 15, 7, 9, 16>
          <<<dim3(1, 128, batch_size), dim3(256, 1), 512 * sizeof(uint64_t)>>>(
              in, out, root_of_unity_table, mod, barret_mu, n_inverse);
      C10_CUDA_KERNEL_LAUNCH_CHECK();

      InverseCore<true, 8, 6, 0, 7, 16>
          <<<dim3(128, 1, batch_size), dim3(4, 64), 512 * sizeof(uint64_t)>>>(
              out, out, root_of_unity_table, mod, barret_mu, n_inverse);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
    case 17:
      InverseCore<false, 8, 16, 8, 9, 17>
          <<<dim3(1, 256, batch_size), dim3(256, 1), 512 * sizeof(uint64_t)>>>(
              in, out, root_of_unity_table, mod, barret_mu, n_inverse);
      C10_CUDA_KERNEL_LAUNCH_CHECK();

      InverseCore<false, 8, 7, 4, 4, 17>
          <<<dim3(16, 16, batch_size), dim3(32, 8), 512 * sizeof(uint64_t)>>>(
              in, out, root_of_unity_table, mod, barret_mu, n_inverse);
      C10_CUDA_KERNEL_LAUNCH_CHECK();

      InverseCore<true, 8, 3, 0, 4, 17>
          <<<dim3(256, 1, batch_size), dim3(32, 8), 512 * sizeof(uint64_t)>>>(
              out, out, root_of_unity_table, mod, barret_mu, n_inverse);
      C10_CUDA_KERNEL_LAUNCH_CHECK();

      break;
    default:
      TORCH_INTERNAL_ASSERT(false, "Invalid n_power");
  }
}
} // namespace fhe

namespace at::native {

Tensor ntt_cuda(
    const Tensor& in,
    const Tensor& root_of_unity_table,
    const Scalar& mod,
    const Tensor& barret_mu) {
  Tensor out = at::empty_like(in);

  AT_DISPATCH_V2(
      in.scalar_type(),
      "ntt_cuda",
      AT_WRAP([&]() {
        auto in_ptr = reinterpret_cast<uint64_t*>(in.data_ptr<uint64_t>());
        auto out_ptr = reinterpret_cast<uint64_t*>(out.data_ptr<uint64_t>());
        auto root_of_unity_table_ptr = reinterpret_cast<uint64_t*>(
            root_of_unity_table.data_ptr<uint64_t>());
        auto barret_mu_ptr =
            reinterpret_cast<uint64_t*>(barret_mu.data_ptr<uint64_t>());

        auto N = in.numel();
        TORCH_INTERNAL_ASSERT(
            (N == 1 << 15) || (N == 1 << 16) || (N == 1 << 17));
        fhe::NTT(
            16,
            in_ptr,
            out_ptr,
            root_of_unity_table_ptr,
            mod.toUInt64(),
            barret_mu_ptr);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }),
      kUInt64);

  return out;
}

Tensor intt_cuda(
    const Tensor& in,
    const Tensor& root_of_unity_table,
    const Scalar& mod,
    const Tensor& barret_mu,
    const Scalar& n_inverse) {
  Tensor out = at::empty_like(in);

  AT_DISPATCH_V2(
      in.scalar_type(),
      "intt_cuda",
      AT_WRAP([&]() {
        auto in_ptr = reinterpret_cast<uint64_t*>(in.data_ptr<uint64_t>());
        auto out_ptr = reinterpret_cast<uint64_t*>(out.data_ptr<uint64_t>());
        auto root_of_unity_table_ptr = reinterpret_cast<uint64_t*>(
            root_of_unity_table.data_ptr<uint64_t>());
        auto barret_mu_ptr =
            reinterpret_cast<uint64_t*>(barret_mu.data_ptr<uint64_t>());

        auto N = in.numel();
        TORCH_INTERNAL_ASSERT(
            (N == 1 << 15) || (N == 1 << 16) || (N == 1 << 17));
        fhe::InverseNTT(
            16,
            in_ptr,
            out_ptr,
            root_of_unity_table_ptr,
            mod.toUInt64(),
            barret_mu_ptr,
            n_inverse.toUInt64());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }),
      kUInt64);
  return out;
}

} // namespace at::native