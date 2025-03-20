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

#include <algorithm>
#include <vector>

#define MAX_64BIT_VALUE 9223372036854775295
#define MAX_BITS_IN_WORD 61
#define DTYPE double
#define DTYPE2 double2
#define MAKE_DTYPE2 make_double2

namespace fhe {
__device__ DTYPE2 mul(DTYPE2 a, DTYPE2 b) {
  return MAKE_DTYPE2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__global__ void convert_and_pad_inverse(
    DTYPE* inverse,
    DTYPE2* inverse_internal,
    int inverse_size) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  DTYPE value = (tid < inverse_size) ? inverse[tid] : 0.0;
  inverse_internal[tid] = MAKE_DTYPE2(value, 0.0);
}

__global__ void fft_stage_kernel(
    DTYPE2* vals,
    int num_stages,
    int vals_size,
    int m_M,
    int64_t* m_rotGroup,
    DTYPE2* m_ksiPows) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total_threads = vals_size / 2;
  if (tid >= total_threads)
    return;
  for (int s = 0; s < num_stages; ++s) {
    int len_size = vals_size >> s;
    int len_h = len_size >> 1;
    int len_q = len_size << 2;
    int gap = m_M / len_q;

    int block_idx = tid / len_h;
    int j = tid % len_h;
    int i = block_idx * len_size;

    int rot = m_rotGroup[j] % len_q;
    int idx = (len_q - rot) * gap;

    DTYPE2 val_low = vals[i + j];
    DTYPE2 val_high = vals[i + j + len_h];
    DTYPE2 u = MAKE_DTYPE2(val_low.x + val_high.x, val_low.y + val_high.y);
    DTYPE2 v = MAKE_DTYPE2(val_low.x - val_high.x, val_low.y - val_high.y);
    vals[i + j] = u;
    vals[i + j + len_h] = mul(v, m_ksiPows[idx]);
    __syncthreads();
  }
}

__global__ void bit_reverse_normalize_kernel(
    DTYPE2* vals,
    int n,
    int num_bits) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n)
    return;

  int reversed = 0;
  for (int i = 0; i < num_bits; ++i) {
    reversed <<= 1;
    reversed |= (tid >> i) & 1;
  }
  if (reversed > tid) {
    DTYPE2 temp = vals[tid];
    vals[tid] = vals[reversed];
    vals[reversed] = temp;
  }
}
__global__ void normalize_kernel(DTYPE2* vals, int n, DTYPE factor) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n)
    return;
  vals[tid].x *= factor;
  vals[tid].y *= factor;
}

__global__ void scaleAndCheckOverflow(
    DTYPE2* inverse,
    int slots,
    DTYPE scaling_factor,
    int64_t* temp,
    int64_t* log_approx_out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= slots)
    return;
  DTYPE2 val = inverse[i];
  val.x *= scaling_factor;
  val.y *= scaling_factor;

  DTYPE abs_real = fabs(val.x);
  DTYPE abs_imag = fabs(val.y);
  int logc = 0;
  if (((abs_imag - 0) < 1e-6) && ((abs_real - 0) < 1e-6)) {
    logc = 0;
  } else {
    logc = max((int)ceil(log2(abs_real)), (int)ceil(log2(abs_imag)));
  }
  if (logc < 0) {
    printf("Too small scaling factor\n");
    return;
  }
  int log_valid = min(logc, MAX_BITS_IN_WORD);
  int log_approx = logc - log_valid;
  log_approx_out[0] = log_approx;
  int approx_factor = 1 << log_approx;
  DTYPE dre = val.x / approx_factor;
  DTYPE dim = val.y / approx_factor;

  if (abs(dre) > MAX_64BIT_VALUE || abs(dim) > MAX_64BIT_VALUE) {
    printf(
        "Overflow in data encoding - scaled input is too large to fit into a NativeInteger (60 bits).\n"
        "Try decreasing scaling factor.");
    return;
  }

  long long re = llround(dre);
  long long im = llround(dim);

  temp[i] = (re < 0) ? (MAX_64BIT_VALUE + re) : re;
  temp[i + slots] = (im < 0) ? (MAX_64BIT_VALUE + im) : im;
}

__global__ void fit_to_native_vector_kernel(
    int64_t* vec,
    int64_t big_bound,
    uint64_t* native_vec,
    uint64_t* native_modulus,
    int N,
    int dslots,
    int gap) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  const int l = i / dslots;
  const int slot = i % dslots;
  const int64_t bigValueHf = big_bound >> 1;
  const int64_t diff = big_bound - native_modulus[l];
  const int64_t n = vec[slot];

  uint64_t result = n > bigValueHf ? ((n - diff) % native_modulus[l])
                                   : (n % native_modulus[l]);
  native_vec[l * N + gap * slot] = result;
}

__global__ void mul_mod_kernel(
    uint64_t* encoded_vector,
    uint64_t* crt_approx,
    uint64_t* moduliQ,
    int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int l = idx / N;
  encoded_vector[idx] = (encoded_vector[idx] * crt_approx[l]) %
      moduliQ[l]; // use utils mul_mod, maybe better
}

} // namespace fhe

namespace at::native {
static std::vector<uint64_t> crt_mult(
    const std::vector<uint64_t>& a,
    const std::vector<uint64_t>& b,
    const uint64_t* moduli) {
  std::vector<uint64_t> result(a.size());
  for (size_t i = 0; i < a.size(); i++) {
    result[i] = (a[i] * b[i]) % moduli[i];
  }
  return result;
}

static void convert_inverse(
    DTYPE* inverse,
    DTYPE2* inverse_internal,
    int inverse_size,
    int slots) {
  const int blockSize = 256;
  const int gridSize = (slots + blockSize - 1) / blockSize;
  auto stream = at::cuda::getCurrentCUDAStream();
  fhe::convert_and_pad_inverse<<<gridSize, blockSize, 0, stream>>>(
      inverse, inverse_internal, inverse_size);
}

static void fft_special_inv_cuda(
    DTYPE2* inverse,
    int64_t* precompute_rotgroups,
    DTYPE2* precompute_ksipows,
    int64_t M,
    int vals_size) {
  dim3 block(256);
  auto stream = at::cuda::getCurrentCUDAStream();
  int total_threads = vals_size / 2;
  dim3 grid((total_threads + block.x - 1) / block.x);

  fhe::fft_stage_kernel<<<grid, block, 0, stream>>>(
      inverse,
      log2(vals_size),
      vals_size,
      M,
      precompute_rotgroups,
      precompute_ksipows);
  cudaDeviceSynchronize();

  dim3 grid_br((vals_size + block.x - 1) / block.x);
  DTYPE factor = 1.0f / vals_size;
  fhe::bit_reverse_normalize_kernel<<<grid_br, block, 0, stream>>>(
      inverse, vals_size, (int)log2f(vals_size));
  cudaDeviceSynchronize();
  fhe::normalize_kernel<<<grid_br, block, 0, stream>>>(
      inverse, vals_size, factor);
  cudaDeviceSynchronize();
}

void fit_to_native_vector(
    int64_t* d_vec,
    int64_t big_bound,
    uint64_t* d_native_vec,
    uint64_t* native_modulus,
    int dslots,
    int gap,
    int cur_limbs,
    int N) {
  if (cur_limbs == 0)
    return;
  auto stream = at::cuda::getCurrentCUDAStream();
  const int blockSize = 256;
  const int gridSize = (dslots * cur_limbs + blockSize - 1) / blockSize;
  fhe::fit_to_native_vector_kernel<<<gridSize, blockSize, 0, stream>>>(
      d_vec, big_bound, d_native_vec, native_modulus, N, dslots, gap);
}

void scale_noise_degree_vector(
    uint64_t* elements_ptr,
    uint64_t* primes_ptr,
    uint64_t* moduli,
    int64_t cur_limbs,
    int64_t N,
    int64_t noise_scale_deg,
    DTYPE scaling_factor) {
  DTYPE pow_p = scaling_factor;
  std::vector<uint64_t> crt_pow_p(cur_limbs, llround(pow_p));
  auto curr_pow_p = crt_pow_p;
  for (int i = 2; i < noise_scale_deg; i++) {
    curr_pow_p = crt_mult(curr_pow_p, crt_pow_p, moduli);
  }
  uint64_t* d_curr_pow_p;
  cudaMalloc(&d_curr_pow_p, sizeof(uint64_t) * curr_pow_p.size());
  cudaMemcpy(
      curr_pow_p.data(),
      d_curr_pow_p,
      sizeof(uint64_t) * curr_pow_p.size(),
      cudaMemcpyHostToDevice);
  auto stream = at::cuda::getCurrentCUDAStream();
  int threadsPerBlock = 256;
  int blocksPerGrid = (cur_limbs * N + threadsPerBlock - 1) / threadsPerBlock;
  fhe::mul_mod_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
      elements_ptr, d_curr_pow_p, primes_ptr, N);
}

void scale_log_approx_vector(
    uint64_t* elements_ptr,
    uint64_t* primes_ptr,
    uint64_t* moduli,
    int log_approx,
    int64_t cur_limbs,
    int64_t N) {
  int MAX_LOG_STEP = 60;
  int log_step = std::min(log_approx, MAX_LOG_STEP);
  int int_step = 1 << log_step;

  std::vector<uint64_t> crt_approx(cur_limbs, int_step);
  log_approx -= log_step;

  while (log_approx > 0) {
    log_step = std::min(log_approx, MAX_LOG_STEP);
    int_step = 1 << log_step;

    std::vector<uint64_t> crt_sf(cur_limbs, int_step);
    crt_approx = crt_mult(crt_approx, crt_sf, moduli);
    log_approx -= log_step;
  }
  uint64_t* d_crt_approx_ptr;
  cudaMalloc(&d_crt_approx_ptr, sizeof(uint64_t) * crt_approx.size());
  cudaMemcpy(
      crt_approx.data(),
      d_crt_approx_ptr,
      sizeof(uint64_t) * crt_approx.size(),
      cudaMemcpyHostToDevice);
  auto stream = at::cuda::getCurrentCUDAStream();
  int threadsPerBlock = 256;
  int blocksPerGrid = (cur_limbs * N + threadsPerBlock - 1) / threadsPerBlock;

  fhe::mul_mod_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
      elements_ptr, d_crt_approx_ptr, primes_ptr, N);
}

static void encode_template(
    const Tensor& inverse,
    const Tensor& inverse_internal,
    const Tensor& temp,
    const Tensor& primes,
    const Tensor& precompute_rotgroups,
    const Tensor& precompute_ksipows,
    int64_t M,
    int64_t N,
    int64_t cur_limbs,
    int64_t slots,
    int64_t noise_scale_deg,
    DTYPE scaling_factor,
    const Tensor& power_of_roots_shoup,
    const Tensor& power_of_roots,
    bool use_fft,
    Tensor& res) {
  AT_DISPATCH_V2(
      res.scalar_type(),
      "encode_impl",
      AT_WRAP([&]() {
        auto inverse_ptr = reinterpret_cast<DTYPE*>(inverse.data_ptr<DTYPE>());
        auto inverse_internal_ptr =
            reinterpret_cast<DTYPE2*>(inverse_internal.data_ptr<DTYPE>());
        auto precompute_ksipows_ptr =
            reinterpret_cast<DTYPE2*>(precompute_ksipows.data_ptr<DTYPE>());
        auto rotGroups = reinterpret_cast<int64_t*>(
            precompute_rotgroups.data_ptr<int64_t>());
        auto elements_ptr =
            reinterpret_cast<uint64_t*>(res.data_ptr<uint64_t>());
        auto primes_ptr =
            reinterpret_cast<uint64_t*>(primes.data_ptr<uint64_t>());
        auto temp_ptr = reinterpret_cast<int64_t*>(temp.data_ptr<int64_t>());
        auto inverse_size = inverse.numel();
        if (use_fft) {
          convert_inverse(inverse_ptr, inverse_internal_ptr, inverse_size, slots);
          fft_special_inv_cuda(
              inverse_internal_ptr,
              rotGroups,
              precompute_ksipows_ptr,
              M,
              slots);
        }
        cudaDeviceSynchronize();
        auto stream = at::cuda::getCurrentCUDAStream();
        const int blockDim2 = 256;
        const int gridDim2 = (slots + blockDim2 - 1) / blockDim2;
        const int temp_size = 2 * slots;
        int64_t* d_log_approx;
        cudaMalloc(&d_log_approx, sizeof(int64_t));
        fhe::scaleAndCheckOverflow<<<gridDim2, blockDim2, 0, stream>>>(
            inverse_internal_ptr,
            slots,
            scaling_factor,
            temp_ptr,
            d_log_approx);
        cudaDeviceSynchronize();
        int gap = N / temp_size;
        fit_to_native_vector(
            temp_ptr,
            MAX_64BIT_VALUE,
            elements_ptr,
            primes_ptr,
            temp_size,
            gap,
            cur_limbs,
            N);
        cudaDeviceSynchronize();
        int* h_log_approx = new int[1];
        cudaMemcpy(
            h_log_approx, d_log_approx, sizeof(int), cudaMemcpyDeviceToHost);
        int log_approx = h_log_approx[0];
        if (noise_scale_deg > 1) {
          //todo: need optimize
          scale_noise_degree_vector(
              elements_ptr,
              primes_ptr,
              primes.cpu().data_ptr<uint64_t>(),
              cur_limbs,
              N,
              noise_scale_deg,
              scaling_factor);
        }

        if (log_approx > 0) {
          //todo: need optimize
          scale_log_approx_vector(
              elements_ptr,
              primes_ptr,
              primes.cpu().data_ptr<uint64_t>(),
              log_approx,
              cur_limbs,
              N);
        }
        cudaDeviceSynchronize();
        NTT_impl(
            elements_ptr,
            elements_ptr,
            cur_limbs,
            N,
            power_of_roots_shoup.data_ptr<uint64_t>(),
            primes.data_ptr<uint64_t>(),
            power_of_roots.data_ptr<uint64_t>());
        cudaDeviceSynchronize();
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }),
      kUInt64);
}

Tensor encode_cuda(
    const Tensor& inverse,
    const Tensor& inverse_internal,
    const Tensor& temp,
    const Tensor& primes,
    const Tensor& precompute_rotgroups,
    const Tensor& precompute_ksipows,
    int64_t M,
    int64_t N,
    int64_t cur_limbs,
    int64_t slots,
    int64_t noise_scale_deg,
    DTYPE scaling_factor,
    const Tensor& power_of_roots_shoup,
    const Tensor& power_of_roots,
    bool use_fft) {
  Tensor out = at::zeros({cur_limbs, N}, primes.options());
  encode_template(
      inverse,
      inverse_internal,
      temp,
      primes,
      precompute_rotgroups,
      precompute_ksipows,
      M,
      N,
      cur_limbs,
      slots,
      noise_scale_deg,
      scaling_factor,
      power_of_roots_shoup,
      power_of_roots,
      use_fft,
      out);
  return out;
}
} // namespace at::native