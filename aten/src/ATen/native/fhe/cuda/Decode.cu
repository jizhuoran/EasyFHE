#include <ATen/Dispatch_v2.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/TensorBody.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/empty.h>
#include <ATen/native/fhe/cuda/CommonOperation.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/complex.h>
#include <algorithm>
#include <cmath>

namespace {

static constexpr int kDecodeBlockSize = 256;
static constexpr int kDecodeMaxLimbs = 64;

__device__ __forceinline__ uint64_t mod_mul_device(
    uint64_t lhs,
    uint64_t rhs,
    uint64_t modulus) {
  return static_cast<uint64_t>(
      (static_cast<unsigned __int128>(lhs) * rhs) % modulus);
}

__device__ __forceinline__ double mixed_radix_to_double_device(
    const uint64_t* digits,
    const uint64_t* moduli,
    int64_t limbs) {
  double value = 0.0;
  for (int64_t i = limbs - 1; i >= 0; --i) {
    value = value * static_cast<double>(moduli[i]) + static_cast<double>(digits[i]);
  }
  return value;
}

__device__ __forceinline__ double mixed_radix_fraction_device(
    const uint64_t* digits,
    const uint64_t* moduli,
    int64_t limbs) {
  double value = 0.0;
  // Garner digit i is weighted by the product of all preceding moduli.
  for (int64_t i = 0; i < limbs; ++i) {
    value = (value + static_cast<double>(digits[i])) / static_cast<double>(moduli[i]);
  }
  return value;
}

__device__ __forceinline__ uint32_t reverse_bits_device(uint32_t value, uint32_t width) {
  uint32_t result = 0;
  for (uint32_t i = 0; i < width; ++i) {
    result = (result << 1) | (value & 1U);
    value >>= 1;
  }
  return result;
}

__device__ __forceinline__ uint32_t bit_width_device(uint32_t value) {
  uint32_t width = 0;
  while (value != 0) {
    ++width;
    value >>= 1;
  }
  return width;
}

__device__ __forceinline__ double centered_crt_to_double_device(
    const uint64_t* coeffs,
    int64_t coeff_idx,
    int64_t ring_dim,
    const uint64_t* moduli,
    const uint64_t* crt_inv_moduli,
    int64_t limbs) {
  uint64_t digits[kDecodeMaxLimbs];
#pragma unroll
  for (int64_t i = 0; i < kDecodeMaxLimbs; ++i) {
    if (i < limbs) {
      digits[i] = 0;
    }
  }

  for (int64_t i = 0; i < limbs; ++i) {
    uint64_t value = coeffs[i * ring_dim + coeff_idx] % moduli[i];
    for (int64_t j = 0; j < i; ++j) {
      const uint64_t digit = digits[j] % moduli[i];
      value = (value >= digit) ? (value - digit) : (value + moduli[i] - digit);
      value = mod_mul_device(value, crt_inv_moduli[i * limbs + j], moduli[i]);
    }
    digits[i] = value;
  }

  if (mixed_radix_fraction_device(digits, moduli, limbs) <= 0.5) {
    return mixed_radix_to_double_device(digits, moduli, limbs);
  }

  uint64_t complement[kDecodeMaxLimbs];
  uint64_t borrow = 0;
  for (int64_t i = 0; i < limbs; ++i) {
    const uint64_t digit = digits[i];
    const uint64_t modulus = moduli[i];
    if (digit == 0 && borrow == 0) {
      complement[i] = 0;
      borrow = 0;
    } else {
      complement[i] = modulus - digit - borrow;
      borrow = 1;
    }
  }
  return -mixed_radix_to_double_device(complement, moduli, limbs);
}

__global__ void decode_crt_slots_kernel(
    const uint64_t* __restrict__ coeffs,
    const uint64_t* __restrict__ moduli,
    const uint64_t* __restrict__ crt_inv_moduli,
    c10::complex<double>* __restrict__ cur_values,
    int64_t ring_dim,
    int64_t limbs,
    int64_t slots,
    int64_t gap,
    int64_t nh,
    double scaling_pre) {
  const int64_t slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= slots) {
    return;
  }
  const int64_t idx = slot * gap;
  const double real = centered_crt_to_double_device(
      coeffs, idx, ring_dim, moduli, crt_inv_moduli, limbs);
  const double imag = centered_crt_to_double_device(
      coeffs, idx + nh, ring_dim, moduli, crt_inv_moduli, limbs);
  cur_values[slot] = c10::complex<double>(real * scaling_pre, imag * scaling_pre);
}

__global__ void combine_conjugate_bitrev_kernel(
    const c10::complex<double>* __restrict__ cur_values,
    c10::complex<double>* __restrict__ work,
    int64_t slots,
    double scale) {
  const int64_t slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= slots) {
    return;
  }

  c10::complex<double> conjugate(0.0, 0.0);
  if (slot == 0) {
    conjugate = c10::complex<double>(cur_values[0].real(), -cur_values[0].imag());
  } else {
    const auto source = cur_values[slots - slot];
    conjugate = c10::complex<double>(-source.imag(), -source.real());
  }
  const auto value = c10::complex<double>(
      scale * (cur_values[slot].real() + conjugate.real()),
      scale * (cur_values[slot].imag() + conjugate.imag()));
  work[slot] = value;
}

__global__ void bitrev_copy_kernel(
    const c10::complex<double>* __restrict__ input,
    c10::complex<double>* __restrict__ output,
    int64_t slots) {
  const int64_t slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= slots) {
    return;
  }
  const uint32_t width = bit_width_device(static_cast<uint32_t>(slots - 1));
  output[slot] = input[reverse_bits_device(static_cast<uint32_t>(slot), width)];
}

__global__ void fft_special_stage_kernel(
    c10::complex<double>* __restrict__ values,
    const uint32_t* __restrict__ rot_group,
    const c10::complex<double>* __restrict__ ksi_pows,
    int64_t slots,
    int64_t cycl_order,
    int64_t len_size) {
  const int64_t len_h = len_size >> 1;
  const int64_t total = slots >> 1;
  const int64_t rem = blockIdx.x * blockDim.x + threadIdx.x;
  if (rem >= total) {
    return;
  }
  const int64_t group = rem / len_h;
  const int64_t j = rem - group * len_h;
  const int64_t base = group * len_size + j;
  const int64_t len_q = len_size << 2;
  const int64_t gap = cycl_order / len_q;
  const int64_t root_index =
      static_cast<int64_t>(rot_group[j] % static_cast<uint32_t>(len_q)) * gap;

  const auto u = values[base];
  const auto v = values[base + len_h] * ksi_pows[root_index];
  values[base] = u + v;
  values[base + len_h] = u - v;
}

__global__ void real_output_kernel(
    const c10::complex<double>* __restrict__ values,
    double* __restrict__ output,
    int64_t slots) {
  const int64_t slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= slots) {
    return;
  }
  output[slot] = values[slot].real();
}

} // namespace

namespace at::native {

Tensor ckks_decode_phase_cuda(
    const Tensor& phase,
    const Tensor& moduli_q,
    const Tensor& crt_inv_moduli,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& rot_group,
    const Tensor& ksi_pows,
    int64_t cur_limbs,
    int64_t plaintext_modulus_bits,
    int64_t noise_scale_deg,
    int64_t slots) {
  TORCH_CHECK(phase.is_cuda(), "phase must be CUDA");
  TORCH_CHECK(phase.scalar_type() == at::kUInt64, "phase must be uint64");
  TORCH_CHECK(phase.dim() == 2, "phase must be [limbs, N]");
  TORCH_CHECK(cur_limbs > 0 && cur_limbs <= phase.size(0), "invalid cur_limbs");
  TORCH_CHECK(cur_limbs <= kDecodeMaxLimbs, "ckks_decode_phase_cuda supports at most ", kDecodeMaxLimbs, " limbs");
  TORCH_CHECK(moduli_q.is_cuda() && crt_inv_moduli.is_cuda(), "CRT inputs must be CUDA tensors");
  TORCH_CHECK(inverse_power_of_roots_div_two.is_cuda() && inverse_scaled_power_of_roots_div_two.is_cuda(), "iNTT tables must be CUDA tensors");
  TORCH_CHECK(rot_group.is_cuda() && ksi_pows.is_cuda(), "FFT tables must be CUDA tensors");
  TORCH_CHECK(moduli_q.scalar_type() == at::kUInt64, "moduli_q must be uint64");
  TORCH_CHECK(crt_inv_moduli.scalar_type() == at::kUInt64, "crt_inv_moduli must be uint64");
  TORCH_CHECK(rot_group.scalar_type() == at::kUInt32, "rot_group must be uint32");
  TORCH_CHECK(ksi_pows.scalar_type() == at::kComplexDouble, "ksi_pows must be complex128");

  const int64_t ring_dim = phase.size(1);
  TORCH_CHECK(ring_dim > 0 && (ring_dim & (ring_dim - 1)) == 0, "ring_dim must be a power of two");
  const int64_t nh = ring_dim >> 1;
  slots = slots == 0 ? nh : slots;
  TORCH_CHECK(slots > 0 && slots <= nh && nh % slots == 0, "invalid CKKS slots");
  TORCH_CHECK((slots & (slots - 1)) == 0, "slots must be a power of two");
  TORCH_CHECK(rot_group.numel() >= slots, "rot_group table is too small");
  TORCH_CHECK(moduli_q.numel() >= cur_limbs, "not enough Q moduli");
  TORCH_CHECK(crt_inv_moduli.numel() >= cur_limbs * cur_limbs, "CRT inverse table is too small");

  Tensor phase_contig = phase.contiguous();
  Tensor coeffs = at::empty_like(phase_contig);
  iNTT_impl(
      coeffs.data_ptr<uint64_t>(),
      phase_contig.data_ptr<uint64_t>(),
      static_cast<size_t>(cur_limbs),
      static_cast<size_t>(ring_dim),
      1,
      1,
      1,
      1,
      moduli_q.data_ptr<uint64_t>(),
      inverse_power_of_roots_div_two.data_ptr<uint64_t>(),
      inverse_scaled_power_of_roots_div_two.data_ptr<uint64_t>());

  Tensor cur_values = at::empty({slots}, phase.options().dtype(at::kComplexDouble));
  Tensor combined = at::empty({slots}, phase.options().dtype(at::kComplexDouble));
  Tensor work = at::empty({slots}, phase.options().dtype(at::kComplexDouble));
  Tensor output = at::empty({slots}, phase.options().dtype(at::kDouble));

  auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 blocks((slots + kDecodeBlockSize - 1) / kDecodeBlockSize);
  const double scaling_pre = std::pow(
      2.0,
      -static_cast<double>(plaintext_modulus_bits) *
          static_cast<double>(std::max<int64_t>(noise_scale_deg - 1, 0)));
  decode_crt_slots_kernel<<<blocks, kDecodeBlockSize, 0, stream>>>(
      coeffs.data_ptr<uint64_t>(),
      moduli_q.data_ptr<uint64_t>(),
      crt_inv_moduli.data_ptr<uint64_t>(),
      cur_values.data_ptr<c10::complex<double>>(),
      ring_dim,
      cur_limbs,
      slots,
      nh / slots,
      nh,
      scaling_pre);

  const double scale = 0.5 * std::pow(2.0, -static_cast<double>(plaintext_modulus_bits));
  combine_conjugate_bitrev_kernel<<<blocks, kDecodeBlockSize, 0, stream>>>(
      cur_values.data_ptr<c10::complex<double>>(),
      combined.data_ptr<c10::complex<double>>(),
      slots,
      scale);
  bitrev_copy_kernel<<<blocks, kDecodeBlockSize, 0, stream>>>(
      combined.data_ptr<c10::complex<double>>(),
      work.data_ptr<c10::complex<double>>(),
      slots);

  const int64_t cycl_order = ring_dim * 2;
  for (int64_t len_size = 2; len_size <= slots; len_size <<= 1) {
    const int64_t total = slots >> 1;
    const dim3 stage_blocks((total + kDecodeBlockSize - 1) / kDecodeBlockSize);
    fft_special_stage_kernel<<<stage_blocks, kDecodeBlockSize, 0, stream>>>(
        work.data_ptr<c10::complex<double>>(),
        rot_group.data_ptr<uint32_t>(),
        ksi_pows.data_ptr<c10::complex<double>>(),
        slots,
        cycl_order,
        len_size);
  }
  real_output_kernel<<<blocks, kDecodeBlockSize, 0, stream>>>(
      work.data_ptr<c10::complex<double>>(),
      output.data_ptr<double>(),
      slots);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

} // namespace at::native
