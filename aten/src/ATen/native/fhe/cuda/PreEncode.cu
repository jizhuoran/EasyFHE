#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/interned_strings.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/thread_constants.h>
#include <ATen/ops/empty.h>
#include <c10/util/complex.h>
#include <cmath>
#include <vector>

namespace fhe {
static constexpr int kPreEncodeBlockSize = 256;
static constexpr int kPreEncodeSharedMaxSlots = 2048;
static constexpr int kPreEncodeMaxLargeTiles = 32;
static constexpr int kAbsMaxBlockSize = 256;

template <typename scalar_t>
__device__ __forceinline__ c10::complex<double> to_complex_double(
    const scalar_t& value) {
  return c10::complex<double>(
      static_cast<double>(value.real()),
      static_cast<double>(value.imag()));
}

__device__ __forceinline__ double warp_reduce_max(double value) {
#pragma unroll
  for (int offset = C10_WARP_SIZE >> 1; offset > 0; offset >>= 1) {
    value = fmax(value, __shfl_down_sync(0xffffffff, value, offset));
  }
  return value;
}

__device__ __forceinline__ double block_reduce_max(double value) {
  __shared__ double shared[kAbsMaxBlockSize / C10_WARP_SIZE];
  const int lane = threadIdx.x % C10_WARP_SIZE;
  const int warp = threadIdx.x / C10_WARP_SIZE;
  value = warp_reduce_max(value);
  if (lane == 0) {
    shared[warp] = value;
  }
  __syncthreads();

  value = threadIdx.x < blockDim.x / C10_WARP_SIZE ? shared[lane] : 0.0;
  if (warp == 0) {
    value = warp_reduce_max(value);
  }
  return value;
}

__global__ void absmax_final_kernel(
    const double* partials,
    double* output,
    int64_t numel) {
  double local_max = 0.0;
  for (int64_t i = threadIdx.x; i < numel; i += blockDim.x) {
    local_max = fmax(local_max, partials[i]);
  }

  const double block_max = block_reduce_max(local_max);
  if (threadIdx.x == 0) {
    output[0] = block_max;
  }
}

template <typename scalar_t>
__global__ void pre_encode_large_stage_kernel(
    const scalar_t* input,
    c10::complex<double>* workspace,
    const uint32_t* rot_group,
    const c10::complex<double>* ksi_pows,
    int64_t slots,
    int64_t M,
    int64_t tiles_per_row) {
  const int64_t tile_size = kPreEncodeSharedMaxSlots;
  const int64_t r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= tile_size) {
    return;
  }
  const int64_t row = blockIdx.y;
  const int64_t row_base = row * slots;

  c10::complex<double> values[kPreEncodeMaxLargeTiles];
#pragma unroll
  for (int64_t t = 0; t < kPreEncodeMaxLargeTiles; ++t) {
    values[t] = c10::complex<double>(0.0, 0.0);
  }
  for (int64_t t = 0; t < tiles_per_row; ++t) {
    values[t] = to_complex_double(input[row_base + t * tile_size + r]);
  }

  for (int64_t tile_group_size = tiles_per_row; tile_group_size > 1;
       tile_group_size >>= 1) {
    const int64_t half_tiles = tile_group_size >> 1;
    const int64_t len_size = tile_size * tile_group_size;
    const int64_t len_q = len_size << 2;
    const int64_t gap = M / len_q;

    for (int64_t group_tile = 0; group_tile < tiles_per_row;
         group_tile += tile_group_size) {
      for (int64_t q_tile = 0; q_tile < half_tiles; ++q_tile) {
        const int64_t j = q_tile * tile_size + r;
        const uint32_t rot = rot_group[j] % static_cast<uint32_t>(len_q);
        const int64_t root_index = (len_q - rot) * gap;
        const auto left = values[group_tile + q_tile];
        const auto right = values[group_tile + q_tile + half_tiles];
        const auto root = ksi_pows[root_index];
        values[group_tile + q_tile] = left + right;
        values[group_tile + q_tile + half_tiles] = (left - right) * root;
      }
    }
  }

  for (int64_t t = 0; t < tiles_per_row; ++t) {
    workspace[row_base + t * tile_size + r] = values[t];
  }
}

template <typename scalar_t>
__global__ void pre_encode_stage1_shared_kernel(
    const scalar_t* input,
    const uint32_t* rot_group,
    const c10::complex<double>* ksi_pows,
    const uint32_t* bitrev,
    double* output,
    double* absmax_partials,
    int64_t slots,
    int64_t M) {
  extern __shared__ unsigned char shared_bytes[];
  auto workspace = reinterpret_cast<c10::complex<double>*>(shared_bytes);
  const int64_t row = blockIdx.x;
  const int64_t row_base = row * slots;

  for (int64_t i = threadIdx.x; i < slots; i += blockDim.x) {
    workspace[i] = to_complex_double(input[row_base + i]);
  }
  __syncthreads();

  for (int64_t len_size = slots; len_size >= 2; len_size >>= 1) {
    const int64_t len_h = len_size >> 1;
    const int64_t len_q = len_size << 2;
    const int64_t gap = M / len_q;
    const int64_t total_work = slots >> 1;

    for (int64_t rem = threadIdx.x; rem < total_work; rem += blockDim.x) {
      const int64_t group = rem / len_h;
      const int64_t j = rem - group * len_h;
      const int64_t base = group * len_size + j;

      const uint32_t rot = rot_group[j] % static_cast<uint32_t>(len_q);
      const int64_t root_index = (len_q - rot) * gap;
      const auto left = workspace[base];
      const auto right = workspace[base + len_h];
      const auto root = ksi_pows[root_index];
      workspace[base] = left + right;
      workspace[base + len_h] = (left - right) * root;
    }
    __syncthreads();
  }

  const double inv_slots = 1.0 / static_cast<double>(slots);
  double local_absmax = 0.0;
  for (int64_t i = threadIdx.x; i < slots; i += blockDim.x) {
    const auto value = workspace[bitrev[i]] * inv_slots;
    const int64_t out_base = row * (2 * slots) + 2 * i;
    output[out_base] = value.real();
    output[out_base + 1] = value.imag();
    local_absmax = fmax(
        local_absmax,
        fmax(fabs(value.real()), fabs(value.imag())));
  }

  const double block_max = block_reduce_max(local_absmax);
  if (threadIdx.x == 0) {
    absmax_partials[row] = block_max;
  }
}

__global__ void pre_encode_stage1_tile_kernel(
    const c10::complex<double>* workspace_global,
    const uint32_t* rot_group,
    const c10::complex<double>* ksi_pows,
    const uint32_t* bitrev,
    double* output,
    double* absmax_partials,
    int64_t slots,
    int64_t M) {
  extern __shared__ unsigned char shared_bytes[];
  auto workspace = reinterpret_cast<c10::complex<double>*>(shared_bytes);
  const int64_t tile_size = kPreEncodeSharedMaxSlots;
  const int64_t tile = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int64_t tile_base = tile * tile_size;
  const int64_t row_base = row * slots;

  for (int64_t i = threadIdx.x; i < tile_size; i += blockDim.x) {
    workspace[i] = workspace_global[row_base + tile_base + i];
  }
  __syncthreads();

  for (int64_t len_size = tile_size; len_size >= 2; len_size >>= 1) {
    const int64_t len_h = len_size >> 1;
    const int64_t len_q = len_size << 2;
    const int64_t gap = M / len_q;
    const int64_t total_work = tile_size >> 1;

    for (int64_t rem = threadIdx.x; rem < total_work; rem += blockDim.x) {
      const int64_t group = rem / len_h;
      const int64_t j = rem - group * len_h;
      const int64_t base = group * len_size + j;

      const uint32_t rot = rot_group[j] % static_cast<uint32_t>(len_q);
      const int64_t root_index = (len_q - rot) * gap;
      const auto left = workspace[base];
      const auto right = workspace[base + len_h];
      const auto root = ksi_pows[root_index];
      workspace[base] = left + right;
      workspace[base + len_h] = (left - right) * root;
    }
    __syncthreads();
  }

  const double inv_slots = 1.0 / static_cast<double>(slots);
  double local_absmax = 0.0;
  for (int64_t i = threadIdx.x; i < tile_size; i += blockDim.x) {
    const int64_t source = tile_base + i;
    const int64_t output_index = bitrev[source];
    const auto value = workspace[i] * inv_slots;
    const int64_t out_base = row * (2 * slots) + 2 * output_index;
    output[out_base] = value.real();
    output[out_base + 1] = value.imag();
    local_absmax = fmax(
        local_absmax,
        fmax(fabs(value.real()), fabs(value.imag())));
  }

  const double block_max = block_reduce_max(local_absmax);
  if (threadIdx.x == 0) {
    absmax_partials[row * gridDim.x + tile] = block_max;
  }
}
} // namespace fhe

namespace at::native {

std::vector<Tensor> fhe_pre_encode_stage1_cuda(
    const Tensor& input,
    int64_t slots,
    int64_t M,
    const Tensor& rotGroup,
    const Tensor& ksiPows,
    const Tensor& bitrev) {
  TORCH_INTERNAL_ASSERT(input.is_cuda(), "fhe_pre_encode_stage1_cuda expects CUDA input");
  TORCH_INTERNAL_ASSERT(
      input.scalar_type() == at::kComplexHalf ||
          input.scalar_type() == at::kComplexFloat ||
          input.scalar_type() == at::kComplexDouble,
      "fhe_pre_encode_stage1_cuda expects complex32, complex64, or complex128 input");
  TORCH_INTERNAL_ASSERT(input.dim() == 1 || input.dim() == 2);
  TORCH_INTERNAL_ASSERT(input.size(input.dim() - 1) == slots);
  TORCH_INTERNAL_ASSERT(rotGroup.is_cuda() && ksiPows.is_cuda() && bitrev.is_cuda());

  Tensor input_2d = input.dim() == 1 ? input.reshape({1, slots}) : input;
  input_2d = input_2d.contiguous();
  const int64_t batch_size = input_2d.size(0);

  Tensor workspace = at::empty(
      {batch_size, slots},
      input_2d.options().dtype(at::kComplexDouble));
  Tensor output = at::empty(
      {batch_size, 2 * slots},
      input_2d.options().dtype(at::kDouble));
  const int64_t tiles_per_row = slots <= fhe::kPreEncodeSharedMaxSlots
      ? 1
      : slots / fhe::kPreEncodeSharedMaxSlots;
  Tensor max_value = at::empty({}, output.options());
  Tensor absmax_partials = at::empty(
      {batch_size * tiles_per_row},
      output.options());

  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_COMPLEX_TYPES_AND(
      at::ScalarType::ComplexHalf,
      input.scalar_type(),
      "fhe_pre_encode_stage1_cuda",
      [&] {
        if (slots <= fhe::kPreEncodeSharedMaxSlots) {
          const size_t shared_bytes =
              static_cast<size_t>(slots) * sizeof(c10::complex<double>);
          fhe::pre_encode_stage1_shared_kernel<<<
              batch_size,
              fhe::kPreEncodeBlockSize,
              shared_bytes,
              stream>>>(
                  input_2d.data_ptr<scalar_t>(),
                  rotGroup.data_ptr<uint32_t>(),
                  ksiPows.data_ptr<c10::complex<double>>(),
                  bitrev.data_ptr<uint32_t>(),
                  output.data_ptr<double>(),
                  absmax_partials.data_ptr<double>(),
                  slots,
                  M);
        } else {
          TORCH_INTERNAL_ASSERT(
              slots % fhe::kPreEncodeSharedMaxSlots == 0,
              "fhe_pre_encode_stage1_cuda expected slots to be a multiple of ",
              fhe::kPreEncodeSharedMaxSlots,
              ", got ",
              slots);
          TORCH_INTERNAL_ASSERT(
              tiles_per_row <= fhe::kPreEncodeMaxLargeTiles,
              "fhe_pre_encode_stage1_cuda supports at most ",
              fhe::kPreEncodeMaxLargeTiles,
              " 2048-slot tiles, got ",
              tiles_per_row);
          fhe::pre_encode_large_stage_kernel<<<
              dim3(
                  (fhe::kPreEncodeSharedMaxSlots +
                   fhe::kPreEncodeBlockSize - 1) /
                      fhe::kPreEncodeBlockSize,
                  batch_size),
              fhe::kPreEncodeBlockSize,
              0,
              stream>>>(
              input_2d.data_ptr<scalar_t>(),
              workspace.data_ptr<c10::complex<double>>(),
              rotGroup.data_ptr<uint32_t>(),
              ksiPows.data_ptr<c10::complex<double>>(),
              slots,
              M,
              tiles_per_row);

          const size_t shared_bytes =
              static_cast<size_t>(fhe::kPreEncodeSharedMaxSlots) *
              sizeof(c10::complex<double>);
          fhe::pre_encode_stage1_tile_kernel<<<
              dim3(tiles_per_row, batch_size),
              fhe::kPreEncodeBlockSize,
              shared_bytes,
              stream>>>(
              workspace.data_ptr<c10::complex<double>>(),
              rotGroup.data_ptr<uint32_t>(),
              ksiPows.data_ptr<c10::complex<double>>(),
              bitrev.data_ptr<uint32_t>(),
              output.data_ptr<double>(),
              absmax_partials.data_ptr<double>(),
              slots,
              M);
        }
      });

  fhe::absmax_final_kernel<<<1, fhe::kAbsMaxBlockSize, 0, stream>>>(
      absmax_partials.data_ptr<double>(),
      max_value.data_ptr<double>(),
      absmax_partials.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, max_value};
}

} // namespace at::native
