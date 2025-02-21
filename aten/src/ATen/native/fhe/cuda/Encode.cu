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

__device__ DTYPE2
complex_mul(DTYPE a_real, DTYPE a_imag, DTYPE b_real, DTYPE b_imag) {
  return MAKE_DTYPE2(
      a_real * b_real - a_imag * b_imag, a_real * b_imag + a_imag * b_real);
}

__global__ void fft_stage_kernel(
    DTYPE* vals_real,
    DTYPE* vals_imag,
    int len_size,
    int vals_size,
    int m_M,
    int64_t* m_rotGroup,
    DTYPE* m_ksiPows_real,
    DTYPE* m_ksiPows_imag) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total_threads = vals_size / 2;

  if (tid >= total_threads)
    return;

  int len_h = len_size >> 1;
  int len_q = len_size << 2;
  int gap = m_M / len_q;

  int block_idx = tid / len_h;
  int j = tid % len_h;
  int i = block_idx * len_size;

  if (i >= 0) {
    int rot = m_rotGroup[j] % len_q;
    int idx = (len_q - rot) * gap;

    DTYPE val_low_real = vals_real[i + j];
    DTYPE val_low_imag = vals_imag[i + j];
    DTYPE val_high_real = vals_real[i + j + len_h];
    DTYPE val_high_imag = vals_imag[i + j + len_h];

    DTYPE u_real = val_low_real + val_high_real;
    DTYPE u_imag = val_low_imag + val_high_imag;
    DTYPE v_real = val_low_real - val_high_real;
    DTYPE v_imag = val_low_imag - val_high_imag;

    DTYPE2 temp =
        complex_mul(v_real, v_imag, m_ksiPows_real[idx], m_ksiPows_imag[idx]);
    v_real = temp.x;
    v_imag = temp.y;

    vals_real[i + j] = u_real;
    vals_imag[i + j] = u_imag;
    vals_real[i + j + len_h] = v_real;
    vals_imag[i + j + len_h] = v_imag;
  }
}

__global__ void bit_reverse_kernel(DTYPE* vals_real, DTYPE* vals_imag, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n)
    return;

  int num_bits = (int)log2f(n);
  int reversed = 0;
  for (int i = 0; i < num_bits; ++i) {
    reversed <<= 1;
    reversed |= (tid >> i) & 1;
  }

  if (reversed > tid) {
    DTYPE temp_real = vals_real[tid];
    DTYPE temp_imag = vals_imag[tid];
    vals_real[tid] = vals_real[reversed];
    vals_imag[tid] = vals_imag[reversed];
    vals_real[reversed] = temp_real;
    vals_imag[reversed] = temp_imag;
  }
}

__global__ void normalize_kernel(
    DTYPE* vals_real,
    DTYPE* vals_imag,
    int n,
    DTYPE factor) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n)
    return;

  vals_real[tid] *= factor;
  vals_imag[tid] *= factor;
}

__global__ void scaleAndCheckOverflow(
    DTYPE* inverse_real,
    DTYPE* inverse_imag,
    int slots,
    DTYPE scaling_factor,
    int64_t* temp,
    int64_t* log_approx_out) {
  /*
   * i is slots
   */
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= slots)
    return;

  // 复数缩放
  //  cuComplex val = inverse[i];
  inverse_real[i] *= scaling_factor; // real part
  inverse_imag[i] *= scaling_factor; // imag part

  DTYPE abs_real = fabs(inverse_real[i]);
  DTYPE abs_imag = fabs(inverse_imag[i]);

  int logc = 0;
  if (abs_real > 0)
    logc = max(logc, (int)ceil(log2(abs_real)));
  if (abs_imag > 0)
    logc = max(logc, (int)ceil(log2(abs_imag)));

  if (logc < 0) {
    printf("Too small scaling factor\n");
    return;
  }
  int log_valid = min(logc, MAX_BITS_IN_WORD);
  int log_approx = logc - log_valid;
  log_approx_out[0] = log_approx;
  int approx_factor = 1 << log_approx;
  DTYPE dre = inverse_real[i] / approx_factor;
  DTYPE dim = inverse_imag[i] / approx_factor;

  if (abs(dre) > MAX_64BIT_VALUE || abs(dim) > MAX_64BIT_VALUE) {
    printf(
        "Overflow in data encoding - scaled input is too large to fit into a NativeInteger (60 bits).\n"
        "Try decreasing scaling factor.");
    return;
  }

  // 量化为整数
  long long re = llround(dre);
  long long im = llround(dim);

  // 处理负数溢出
  temp[i] = (re < 0) ? (MAX_64BIT_VALUE + re) : re;
  temp[i + slots] = (im < 0) ? (MAX_64BIT_VALUE + im) : im;
}

__global__ void fit_to_native_vector_kernel(
    int64_t* vec,
    int64_t big_bound,
    uint64_t* native_vec,
    uint64_t* native_modulus,
    int l,
    int dslots,
    int gap) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= dslots)
    return;

  const int64_t bigValueHf = big_bound >> 1;
  const int64_t diff = big_bound - native_modulus[l];
  const int64_t n = vec[i];

  uint64_t result;
  if (n > bigValueHf) {
    int64_t temp = n - diff;
    result = temp % native_modulus[l];
  } else {
    result = n % native_modulus[l];
  }

  native_vec[gap * i] = result;
}

__global__ void mul_mod_kernel(
    uint64_t* encoded_vector,
    uint64_t* crt_approx,
    uint64_t* moduliQ,
    int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int l = idx / N;
  encoded_vector[idx] = (encoded_vector[idx] * crt_approx[l]) % moduliQ[l];
}

} // namespace fhe

namespace at::native {

static std::vector<uint64_t> crt_mult(
    const std::vector<uint64_t>& a,
    const std::vector<uint64_t>& b,
    const std::vector<uint64_t>& moduli) {
  std::vector<uint64_t> result(a.size());
  for (size_t i = 0; i < a.size(); i++) {
    result[i] = (a[i] * b[i]) % moduli[i]; // 避免溢出
  }
  return result;
}

static void fft_special_inv_cuda(
    DTYPE* inverse_real,
    DTYPE* inverse_imag,
    int64_t* precompute_rotgroups,
    DTYPE* precompute_ksipows_real,
    DTYPE* precompute_ksipows_imag,
    int64_t M,
    int vals_size) {
  int len_size = vals_size;
  dim3 block(256);
  auto stream = at::cuda::getCurrentCUDAStream();
  while (len_size >= 1) {
    int total_threads = vals_size / 2;
    dim3 grid((total_threads + block.x - 1) / block.x);

    fhe::fft_stage_kernel<<<grid, block, 0, stream>>>(
        inverse_real,
        inverse_imag,
        len_size,
        vals_size,
        M,
        precompute_rotgroups,
        precompute_ksipows_real,
        precompute_ksipows_imag);
    cudaDeviceSynchronize();
    len_size >>= 1;
  }

  dim3 grid_br((vals_size + block.x - 1) / block.x);
  fhe::bit_reverse_kernel<<<grid_br, block, 0, stream>>>(
      inverse_real, inverse_imag, vals_size);

  cudaDeviceSynchronize();

  DTYPE factor = 1.0f / vals_size;
  fhe::normalize_kernel<<<grid_br, block, 0, stream>>>(
      inverse_real, inverse_imag, vals_size, factor);
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
  auto stream = at::cuda::getCurrentCUDAStream();
  const int blockSize = 256;
  const int gridSize = (dslots + blockSize - 1) / blockSize;
  for (int i = 0; i < cur_limbs; i++) {
    fhe::fit_to_native_vector_kernel<<<gridSize, blockSize, 0, stream>>>(
        d_vec, big_bound, d_native_vec, native_modulus, i, dslots, gap);
    d_native_vec += N;
  }
}

void scale_noise_degree_vector(
    uint64_t* elements_ptr,
    uint64_t* primes_ptr,
    std::vector<uint64_t>& moduli,
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
    std::vector<uint64_t>& moduli,
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
    const Tensor& inverse_real,
    const Tensor& inverse_imag,
    const Tensor& temp,
    const Tensor& primes,
    const Tensor& precompute_rotgroups,
    const Tensor& precompute_ksipows_real,
    const Tensor& precompute_ksipows_imag,
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
        int inverse_size = inverse_real.numel();
        auto inverse_real_ptr =
            reinterpret_cast<DTYPE*>(inverse_real.data_ptr<DTYPE>());
        auto inverse_imag_ptr =
            reinterpret_cast<DTYPE*>(inverse_imag.data_ptr<DTYPE>());
        auto precompute_ksipows_real_ptr =
            reinterpret_cast<DTYPE*>(precompute_ksipows_real.data_ptr<DTYPE>());
        auto precompute_ksipows_imag_ptr =
            reinterpret_cast<DTYPE*>(precompute_ksipows_imag.data_ptr<DTYPE>());
        auto rotGroups = reinterpret_cast<int64_t*>(
            precompute_rotgroups.data_ptr<int64_t>());
        auto elements_ptr =
            reinterpret_cast<uint64_t*>(res.data_ptr<uint64_t>());
        auto primes_ptr =
            reinterpret_cast<uint64_t*>(primes.data_ptr<uint64_t>());
        if (use_fft) {
          fft_special_inv_cuda(
              inverse_real_ptr,
              inverse_imag_ptr,
              rotGroups,
              precompute_ksipows_real_ptr,
              precompute_ksipows_imag_ptr,
              M,
              inverse_size);
        }

        auto stream = at::cuda::getCurrentCUDAStream();
        const int blockDim2 = 256;
        const int gridDim2 = (inverse_size + blockDim2 - 1) / blockDim2;
        auto temp_ptr = reinterpret_cast<int64_t*>(temp.data_ptr<int64_t>());
        const int temp_size = 2 * slots;
        int64_t* d_log_approx;
        cudaMalloc(&d_log_approx, sizeof(int64_t));
        fhe::scaleAndCheckOverflow<<<gridDim2, blockDim2, 0, stream>>>(
            inverse_real_ptr,
            inverse_imag_ptr,
            slots,
            scaling_factor,
            temp_ptr,
            d_log_approx);

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

        int* h_log_approx = new int[1];
        cudaMemcpy(
            h_log_approx, d_log_approx, sizeof(int), cudaMemcpyDeviceToHost);
        int log_approx = h_log_approx[0];
        std::vector<uint64_t> moduli(cur_limbs, 0);
        cudaMemcpy(
            moduli.data(),
            primes_ptr,
            sizeof(uint64_t) * cur_limbs,
            cudaMemcpyDeviceToHost);

        if (noise_scale_deg > 1) {
          scale_noise_degree_vector(
              elements_ptr,
              primes_ptr,
              moduli,
              cur_limbs,
              N,
              noise_scale_deg,
              scaling_factor);
        }

        if (log_approx > 0) {
          scale_log_approx_vector(
              elements_ptr, primes_ptr, moduli, log_approx, cur_limbs, N);
        }

        NTT_impl(
            elements_ptr,
            elements_ptr,
            cur_limbs,
            N,
            power_of_roots_shoup.data_ptr<uint64_t>(),
            primes.data_ptr<uint64_t>(),
            power_of_roots.data_ptr<uint64_t>());

        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }),
      kUInt64);
}

Tensor encode_cuda(
    const Tensor& res,
    const Tensor& inverse_real,
    const Tensor& inverse_imag,
    const Tensor& temp,
    const Tensor& primes,
    const Tensor& precompute_rotgroups,
    const Tensor& precompute_ksipows_real,
    const Tensor& precompute_ksipows_imag,
    int64_t M,
    int64_t N,
    int64_t cur_limbs,
    int64_t slots,
    int64_t noise_scale_deg,
    DTYPE scaling_factor,
    const Tensor& power_of_roots_shoup,
    const Tensor& power_of_roots,
    bool use_fft) {
  Tensor out = at::empty_like(res);
  out.resize_({cur_limbs, N});
  encode_template(
      inverse_real,
      inverse_imag,
      temp,
      primes,
      precompute_rotgroups,
      precompute_ksipows_real,
      precompute_ksipows_imag,
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

Tensor encode_cuda_(
    Tensor& res,
    const Tensor& inverse_real,
    const Tensor& inverse_imag,
    const Tensor& temp,
    const Tensor& primes,
    const Tensor& precompute_rotgroups,
    const Tensor& precompute_ksipows_real,
    const Tensor& precompute_ksipows_imag,
    int64_t M,
    int64_t N,
    int64_t cur_limbs,
    int64_t slots,
    int64_t noise_scale_deg,
    DTYPE scaling_factor,
    const Tensor& power_of_roots_shoup,
    const Tensor& power_of_roots,
    bool use_fft) {
  res.resize_({cur_limbs, N});
  encode_template(
      inverse_real,
      inverse_imag,
      temp,
      primes,
      precompute_rotgroups,
      precompute_ksipows_real,
      precompute_ksipows_imag,
      M,
      N,
      cur_limbs,
      slots,
      noise_scale_deg,
      scaling_factor,
      power_of_roots_shoup,
      power_of_roots,
      use_fft,
      res);
  return res;
}

Tensor encode_cuda_out(
    const Tensor& res,
    const Tensor& inverse_real,
    const Tensor& inverse_imag,
    const Tensor& temp,
    const Tensor& primes,
    const Tensor& precompute_rotgroups,
    const Tensor& precompute_ksipows_real,
    const Tensor& precompute_ksipows_imag,
    int64_t M,
    int64_t N,
    int64_t cur_limbs,
    int64_t slots,
    int64_t noise_scale_deg,
    DTYPE scaling_factor,
    const Tensor& power_of_roots_shoup,
    const Tensor& power_of_roots,
    bool use_fft,
    Tensor& out) {
  out.resize_({cur_limbs, N});
  encode_template(
      inverse_real,
      inverse_imag,
      temp,
      primes,
      precompute_rotgroups,
      precompute_ksipows_real,
      precompute_ksipows_imag,
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
