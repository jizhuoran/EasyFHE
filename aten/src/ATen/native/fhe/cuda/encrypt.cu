#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <c10/cuda/CUDAGuard.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "ATen/native/fhe/cuda/CommonOperation.h"
#include "ATen/native/fhe/cuda/Utils.cuh"

namespace fhe {
namespace {

constexpr double kGaussianStd = 3.2;
constexpr double kTwoPi = 6.283185307179586476925286766559;
constexpr int kEncryptBlockSize = 256;

__device__ __forceinline__ uint64_t splitmix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

__device__ __forceinline__ double uniform_open01(uint64_t x) {
  return static_cast<double>((x >> 11) + 1) *
      (1.0 / 9007199254740993.0);
}

__device__ __forceinline__ int64_t sample_discrete_gaussian(
    uint64_t seed,
    int sample_id,
    int64_t coeff_idx) {
  const uint64_t base = seed ^
      (static_cast<uint64_t>(sample_id + 1) * 0xd1b54a32d192ed03ULL) ^
      (static_cast<uint64_t>(coeff_idx + 1) * 0x94d049bb133111ebULL);
  const double u1 = uniform_open01(splitmix64(base));
  const double u2 = uniform_open01(splitmix64(base ^ 0xbf58476d1ce4e5b9ULL));
  const double z = sqrt(-2.0 * log(u1)) * cos(kTwoPi * u2);
  return static_cast<int64_t>(llround(kGaussianStd * z));
}

__global__ void sample_gauss_mod_kernel(
    uint64_t* __restrict__ out,
    const uint64_t* __restrict__ primes,
    int64_t l,
    int64_t N,
    uint64_t seed,
    int sample_id) {
  const int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total = l * N;
  if (linear >= total) {
    return;
  }
  const int64_t limb = linear / N;
  const int64_t coeff = linear - limb * N;
  const uint64_t prime = primes[limb];
  const int64_t sample = sample_discrete_gaussian(seed, sample_id, coeff);
  if (sample >= 0) {
    out[linear] = static_cast<uint64_t>(sample) % prime;
  } else {
    const uint64_t magnitude = static_cast<uint64_t>(-sample) % prime;
    out[linear] = magnitude == 0 ? 0 : prime - magnitude;
  }
}

__device__ __forceinline__ uint64_t mul_mod_barrett_128(
    uint64_t a,
    uint64_t b,
    uint64_t prime,
    uint64_t barrett_ratio,
    uint64_t barrett_k) {
  const auto product = mult_64_64_128(a, b);
  return barret_reduction_128_64(product, prime, barrett_ratio, barrett_k);
}

__global__ void encrypt_finish_kernel(
    uint64_t* __restrict__ bx,
    uint64_t* __restrict__ ax,
    const uint64_t* __restrict__ vx,
    const uint64_t* __restrict__ ex0,
    const uint64_t* __restrict__ ex1,
    const uint64_t* __restrict__ ptx,
    const uint64_t* __restrict__ pk0,
    const uint64_t* __restrict__ pk1,
    const uint64_t* __restrict__ primes,
    const uint64_t* __restrict__ barrett_ratio,
    const uint64_t* __restrict__ barrett_k,
    int64_t l,
    int64_t N) {
  const int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total = l * N;
  if (linear >= total) {
    return;
  }
  const int64_t limb = linear / N;
  const uint64_t prime = primes[limb];
  const uint64_t ratio = barrett_ratio[limb];
  const uint64_t k = barrett_k[limb];

  uint64_t bx_value =
      mul_mod_barrett_128(vx[linear], pk0[linear], prime, ratio, k);
  bx_value = add_mod(bx_value, ex0[linear], prime);
  bx_value = add_mod(bx_value, ptx[linear], prime);
  bx[linear] = bx_value;

  uint64_t ax_value =
      mul_mod_barrett_128(vx[linear], pk1[linear], prime, ratio, k);
  ax_value = add_mod(ax_value, ex1[linear], prime);
  ax[linear] = ax_value;
}

void launch_sample(
    uint64_t* out,
    const uint64_t* primes,
    int64_t l,
    int64_t N,
    uint64_t seed,
    int sample_id) {
  const int64_t total = l * N;
  const dim3 block(kEncryptBlockSize);
  const dim3 grid((total + block.x - 1) / block.x);
  auto stream = at::cuda::getCurrentCUDAStream();
  sample_gauss_mod_kernel<<<grid, block, 0, stream>>>(
      out, primes, l, N, seed, sample_id);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_encrypt_finish(
    uint64_t* bx,
    uint64_t* ax,
    const uint64_t* vx,
    const uint64_t* ex0,
    const uint64_t* ex1,
    const uint64_t* ptx,
    const uint64_t* pk0,
    const uint64_t* pk1,
    const uint64_t* primes,
    const uint64_t* barrett_ratio,
    const uint64_t* barrett_k,
    int64_t l,
    int64_t N) {
  const int64_t total = l * N;
  const dim3 block(kEncryptBlockSize);
  const dim3 grid((total + block.x - 1) / block.x);
  auto stream = at::cuda::getCurrentCUDAStream();
  encrypt_finish_kernel<<<grid, block, 0, stream>>>(
      bx,
      ax,
      vx,
      ex0,
      ex1,
      ptx,
      pk0,
      pk1,
      primes,
      barrett_ratio,
      barrett_k,
      l,
      N);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

uint64_t host_seed() {
  const auto now = static_cast<uint64_t>(
      std::chrono::high_resolution_clock::now().time_since_epoch().count());
  return (static_cast<uint64_t>(std::random_device{}()) << 32) ^
      static_cast<uint64_t>(std::random_device{}()) ^ now;
}

} // namespace
} // namespace fhe

namespace at::native {

std::vector<Tensor> encrypt_cuda(
    const Tensor& ptx,
    const Tensor& pk0,
    const Tensor& pk1,
    int64_t l,
    int64_t logn,
    int64_t nh,
    const Tensor& moduliP_scalar,
    const Tensor& moduliQ_scalar,
    const Tensor& primes,
    const Tensor& max_int_diffs,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    const Tensor& power_of_roots_shoup,
    const Tensor& power_of_roots) {
  (void)nh;
  (void)moduliP_scalar;
  (void)moduliQ_scalar;
  (void)max_int_diffs;

  TORCH_CHECK(ptx.is_cuda(), "encrypt_cuda expects CUDA plaintext");
  TORCH_CHECK(pk0.is_cuda() && pk1.is_cuda(), "encrypt_cuda expects CUDA public keys");
  TORCH_CHECK(primes.is_cuda() && barret_ratio.is_cuda() && barret_k.is_cuda(), "encrypt_cuda expects CUDA modulus tables");
  TORCH_CHECK(power_of_roots.is_cuda() && power_of_roots_shoup.is_cuda(), "encrypt_cuda expects CUDA NTT tables");
  TORCH_CHECK(ptx.scalar_type() == at::kUInt64, "encrypt_cuda expects uint64 plaintext");
  TORCH_CHECK(pk0.scalar_type() == at::kUInt64 && pk1.scalar_type() == at::kUInt64, "encrypt_cuda expects uint64 public keys");
  TORCH_CHECK(ptx.dim() == 2 || ptx.dim() == 3, "encrypt_cuda expects [limbs, N] or [1, limbs, N] plaintext");
  if (ptx.dim() == 3) {
    TORCH_CHECK(ptx.size(0) == 1, "encrypt_cuda currently expects batch_size=1");
  }
  TORCH_CHECK(pk0.dim() == 2 && pk1.dim() == 2, "encrypt_cuda expects [limbs, N] public keys");

  const int64_t N = int64_t{1} << logn;
  const int64_t size = l * N;
  TORCH_CHECK(pk0.size(0) >= l && pk1.size(0) >= l, "encrypt_cuda public key has too few limbs");
  TORCH_CHECK(pk0.size(1) == N && pk1.size(1) == N, "encrypt_cuda public key N mismatch");
  TORCH_CHECK(ptx.numel() >= size, "encrypt_cuda plaintext has too few elements");

  const OptionalDeviceGuard device_guard(device_of(ptx));
  const Tensor ptx_contig = ptx.contiguous();
  const Tensor pk0_contig = pk0.narrow(0, 0, l).contiguous();
  const Tensor pk1_contig = pk1.narrow(0, 0, l).contiguous();

  Tensor bx = at::empty({size}, ptx.options());
  Tensor ax = at::empty({size}, ptx.options());
  Tensor vx = at::empty({size}, ptx.options());
  Tensor ex0 = at::empty({size}, ptx.options());
  Tensor ex1 = at::empty({size}, ptx.options());

  const uint64_t seed = fhe::host_seed();
  fhe::launch_sample(vx.mutable_data_ptr<uint64_t>(), primes.data_ptr<uint64_t>(), l, N, seed, 0);
  fhe::launch_sample(ex0.mutable_data_ptr<uint64_t>(), primes.data_ptr<uint64_t>(), l, N, seed, 1);
  fhe::launch_sample(ex1.mutable_data_ptr<uint64_t>(), primes.data_ptr<uint64_t>(), l, N, seed, 2);

  NTT_impl(
      vx.mutable_data_ptr<uint64_t>(),
      l,
      N,
      l,
      1,
      1,
      primes.data_ptr<uint64_t>(),
      power_of_roots_shoup.data_ptr<uint64_t>(),
      power_of_roots.data_ptr<uint64_t>());
  NTT_impl(
      ex0.mutable_data_ptr<uint64_t>(),
      l,
      N,
      l,
      1,
      1,
      primes.data_ptr<uint64_t>(),
      power_of_roots_shoup.data_ptr<uint64_t>(),
      power_of_roots.data_ptr<uint64_t>());
  NTT_impl(
      ex1.mutable_data_ptr<uint64_t>(),
      l,
      N,
      l,
      1,
      1,
      primes.data_ptr<uint64_t>(),
      power_of_roots_shoup.data_ptr<uint64_t>(),
      power_of_roots.data_ptr<uint64_t>());

  fhe::launch_encrypt_finish(
      bx.mutable_data_ptr<uint64_t>(),
      ax.mutable_data_ptr<uint64_t>(),
      vx.data_ptr<uint64_t>(),
      ex0.data_ptr<uint64_t>(),
      ex1.data_ptr<uint64_t>(),
      ptx_contig.data_ptr<uint64_t>(),
      pk0_contig.data_ptr<uint64_t>(),
      pk1_contig.data_ptr<uint64_t>(),
      primes.data_ptr<uint64_t>(),
      barret_ratio.data_ptr<uint64_t>(),
      barret_k.data_ptr<uint64_t>(),
      l,
      N);

  return {bx, ax};
}

} // namespace at::native
