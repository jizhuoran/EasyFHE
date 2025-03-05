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

__global__ void fft_stage_kernel(
    DTYPE2* vals,
    int len_size,
    int vals_size,
    int m_M,
    int64_t* m_rotGroup,
    DTYPE2* m_ksiPows) {
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

  int rot = m_rotGroup[j] % len_q;
  int idx = (len_q - rot) * gap;

  DTYPE2 val_low = vals[i + j];
  DTYPE2 val_high = vals[i + j + len_h];
  DTYPE2 u = MAKE_DTYPE2(val_low.x + val_high.x, val_low.y + val_high.y);
  DTYPE2 v = MAKE_DTYPE2(val_low.x - val_high.x, val_low.y - val_high.y);
  vals[i + j] = u;
  vals[i + j + len_h] = mul(v, m_ksiPows[idx]);
}

__global__ void bit_reverse_kernel(
    DTYPE2* vals,
    int64_t* reserved_order,
    int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n)
    return;
  auto reversed = reserved_order[tid];
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

__global__ void compute_max_logc(
    const DTYPE2* inverse,
    int slots,
    int* max_logc) {
  extern __shared__ int sdata[];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + tid;
  int local_max = 0;

  if (i < slots) {
    DTYPE abs_real = fabs(inverse[i].x);
    DTYPE abs_imag = fabs(inverse[i].y);
    int logc = 0;
    if (abs_real > 0)
      logc = max(logc, (int)ceil(log2(abs_real)));
    if (abs_imag > 0)
      logc = max(logc, (int)ceil(log2(abs_imag)));
    local_max = logc;
  }

  sdata[tid] = local_max;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s)
      sdata[tid] = max(sdata[tid], sdata[tid + s]);
    __syncthreads();
  }

  if (tid == 0)
    atomicMax(max_logc, sdata[0]);
}

__global__ void quantize_values(
    DTYPE2* inverse,
    int64_t* temp,
    DTYPE scaling_factor,
    int log_approx,
    int slots) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= slots)
    return;

  DTYPE re = inverse[i].x * scaling_factor;
  DTYPE im = inverse[i].y * scaling_factor;

  int approx_factor = 1 << log_approx;
  re /= approx_factor;
  im /= approx_factor;

  temp[i] = (llround(re) + MAX_64BIT_VALUE) % MAX_64BIT_VALUE;
  temp[i + slots] = (llround(im) + MAX_64BIT_VALUE) % MAX_64BIT_VALUE;
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

  inverse[i].x *= scaling_factor; // real part
  inverse[i].y *= scaling_factor; // imag part

  DTYPE abs_real = fabs(inverse[i].x);
  DTYPE abs_imag = fabs(inverse[i].y);

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
  DTYPE dre = inverse[i].x / approx_factor;
  DTYPE dim = inverse[i].y / approx_factor;

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
    DTYPE2* inverse,
    int64_t* precompute_rotgroups,
    DTYPE2* precompute_ksipows,
    int64_t* reserved_order,
    int64_t M,
    int vals_size) {
  int len_size = vals_size;
  dim3 block(256);
  auto stream = at::cuda::getCurrentCUDAStream();
  while (len_size >= 1) {
    int total_threads = vals_size / 2;
    dim3 grid((total_threads + block.x - 1) / block.x);

    fhe::fft_stage_kernel<<<grid, block, 0, stream>>>(
        inverse,
        len_size,
        vals_size,
        M,
        precompute_rotgroups,
        precompute_ksipows);
    cudaDeviceSynchronize();
    len_size >>= 1;
  }

  dim3 grid_br((vals_size + block.x - 1) / block.x);
  fhe::bit_reverse_kernel<<<grid_br, block, 0, stream>>>(
      inverse, reserved_order, vals_size);

  cudaDeviceSynchronize();

  DTYPE factor = 1.0f / vals_size;
  fhe::normalize_kernel<<<grid_br, block, 0, stream>>>(
      inverse, vals_size, factor);
  cudaDeviceSynchronize();
}

int scale_and_check_overflow(
    DTYPE2* inverse_ptr,
    int64_t* temp_ptr,
    int slots,
    DTYPE scaling_factor) {
  auto stream = at::cuda::getCurrentCUDAStream();
  int h_logc = 0;
  int* d_logc;
  cudaMalloc(&d_logc, sizeof(int));
  cudaMemset(d_logc, 0, sizeof(int));

  dim3 block(256);
  dim3 grid((slots + block.x - 1) / block.x);
  fhe::compute_max_logc<<<grid, block, block.x * sizeof(int), stream>>>(
      inverse_ptr, slots, d_logc);

  cudaMemcpyAsync(&h_logc, d_logc, sizeof(int), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  // 量化数值
  fhe::quantize_values<<<grid, block, 0, stream>>>(
      inverse_ptr, temp_ptr, scaling_factor, h_logc, slots);
  int log_valid = min(h_logc, MAX_BITS_IN_WORD);
  int log_approx = h_logc - log_valid;
  return log_approx;
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
    const Tensor& inverse,
    const Tensor& temp,
    const Tensor& primes,
    const Tensor& precompute_rotgroups,
    const Tensor& precompute_ksipows,
    const Tensor& precompute_reserve_order,
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
        int inverse_size = inverse.numel() / 2;
        auto inverse_ptr = reinterpret_cast<DTYPE2*>(inverse.data_ptr<DTYPE>());
        auto precompute_ksipows_ptr =
            reinterpret_cast<DTYPE2*>(precompute_ksipows.data_ptr<DTYPE>());
        auto rotGroups = reinterpret_cast<int64_t*>(
            precompute_rotgroups.data_ptr<int64_t>());
        auto elements_ptr =
            reinterpret_cast<uint64_t*>(res.data_ptr<uint64_t>());
        auto primes_ptr =
            reinterpret_cast<uint64_t*>(primes.data_ptr<uint64_t>());
        auto reserved_order = reinterpret_cast<int64_t*>(
            precompute_reserve_order.data_ptr<int64_t>());
        auto temp_ptr = reinterpret_cast<int64_t*>(temp.data_ptr<int64_t>());
        if (use_fft) {
          fft_special_inv_cuda(
              inverse_ptr,
              rotGroups,
              precompute_ksipows_ptr,
              reserved_order,
              M,
              inverse_size);
        }
        auto stream = at::cuda::getCurrentCUDAStream();
        const int blockDim2 = 256;
        const int gridDim2 = (inverse_size + blockDim2 - 1) / blockDim2;
        const int temp_size = 2 * slots;
        int64_t* d_log_approx;
        cudaMalloc(&d_log_approx, sizeof(int64_t));
        fhe::scaleAndCheckOverflow<<<gridDim2, blockDim2, 0, stream>>>(
            inverse_ptr, slots, scaling_factor, temp_ptr, d_log_approx);
        cudaDeviceSynchronize();
//        int log_approx = scale_and_check_overflow(
//            inverse_ptr, temp_ptr, slots, scaling_factor);
//        int temp_size = 2 * slots;
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
    const Tensor& temp,
    const Tensor& primes,
    const Tensor& precompute_rotgroups,
    const Tensor& precompute_ksipows,
    const Tensor& precompute_reserve_order,
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
      temp,
      primes,
      precompute_rotgroups,
      precompute_ksipows,
      precompute_reserve_order,
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