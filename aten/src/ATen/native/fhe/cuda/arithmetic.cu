#include <ATen/Dispatch_v2.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/thread_constants.h>

#include "ATen/native/fhe/cuda/arithmetic.cuh"

#pragma clang diagnostic ignored "-Wmissing-prototypes"

#define BLOCK_SIZE (512)
#define MAX_NUM_BLOCKS (BLOCK_SIZE)

namespace fhe {

__global__ void add_mod_kernel(
    uint64_t* c,
    const uint64_t* a,
    const uint64_t* b,
    uint64_t mod) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  c[i] = add_mod(a[i], b[i], mod);
}

__global__ void add_mod_kernel_(
    uint64_t* self,
    const uint64_t* other,
    uint64_t mod) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  self[i] = add_mod(self[i], other[i], mod);
}

__global__ void sub_mod_kernel(
    uint64_t* c,
    const uint64_t* a,
    const uint64_t* b,
    uint64_t mod) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  c[i] = sub_mod(a[i], b[i], mod);
}

__global__ void sub_mod_kernel_(
    uint64_t* self,
    const uint64_t* other,
    uint64_t mod) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  self[i] = sub_mod(self[i], other[i], mod);
}

__global__ void mul_mod_kernel(
    uint64_t* c,
    const uint64_t* a,
    const uint64_t* b,
    uint64_t mod,
    const uint64_t* barret_mu) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  c[i] = mul_mod(a[i], b[i], mod, barret_mu[0], barret_mu[1]);
}

__global__ void mul_mod_kernel_(
    uint64_t* self,
    const uint64_t* other,
    uint64_t mod,
    const uint64_t* barret_mu) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  self[i] = mul_mod(self[i], other[i], mod, barret_mu[0], barret_mu[1]);
}

} // namespace fhe

namespace at::native {

static void add_mod_template(
    Tensor& c,
    const Tensor& a,
    const Tensor& b,
    uint64_t mod) {
  AT_DISPATCH_V2(
      a.scalar_type(),
      "add_mod_cuda",
      AT_WRAP([&]() {
        auto a_ptr = reinterpret_cast<uint64_t*>(a.data_ptr<uint64_t>());
        auto b_ptr = reinterpret_cast<uint64_t*>(b.data_ptr<uint64_t>());
        auto c_ptr =
            reinterpret_cast<uint64_t*>(c.mutable_data_ptr<uint64_t>());
        auto N = a.numel();
        TORCH_INTERNAL_ASSERT(
            (N == 1 << 15) || (N == 1 << 16) || (N == 1 << 17));
        auto grid = (N + block_work_size() - 1) / block_work_size();
        auto stream = at::cuda::getCurrentCUDAStream();
        fhe::add_mod_kernel<<<grid, block_work_size(), 0, stream>>>(
            c_ptr, a_ptr, b_ptr, mod);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }),
      kUInt64);
}

static void add_mod_template_(Tensor& self, const Tensor& other, uint64_t mod) {
  AT_DISPATCH_V2(
      self.scalar_type(),
      "add_mod_cuda_",
      AT_WRAP([&]() {
        auto self_ptr = reinterpret_cast<uint64_t*>(self.data_ptr<uint64_t>());
        auto other_ptr =
            reinterpret_cast<uint64_t*>(other.data_ptr<uint64_t>());
        auto N = self.numel();
        TORCH_INTERNAL_ASSERT(
            (N == 1 << 15) || (N == 1 << 16) || (N == 1 << 17));
        auto grid = (N + block_work_size() - 1) / block_work_size();
        auto stream = at::cuda::getCurrentCUDAStream();
        fhe::add_mod_kernel_<<<grid, block_work_size(), 0, stream>>>(
            self_ptr, other_ptr, mod);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }),
      kUInt64);
}

static void sub_mod_template(
    Tensor& c,
    const Tensor& a,
    const Tensor& b,
    uint64_t mod) {
  AT_DISPATCH_V2(
      a.scalar_type(),
      "sub_mod_cuda",
      AT_WRAP([&]() {
        auto a_ptr = reinterpret_cast<uint64_t*>(a.data_ptr<uint64_t>());
        auto b_ptr = reinterpret_cast<uint64_t*>(b.data_ptr<uint64_t>());
        auto c_ptr =
            reinterpret_cast<uint64_t*>(c.mutable_data_ptr<uint64_t>());
        auto N = a.numel();
        TORCH_INTERNAL_ASSERT(
            (N == 1 << 15) || (N == 1 << 16) || (N == 1 << 17));
        auto grid = (N + block_work_size() - 1) / block_work_size();
        auto stream = at::cuda::getCurrentCUDAStream();
        fhe::sub_mod_kernel<<<grid, block_work_size(), 0, stream>>>(
            c_ptr, a_ptr, b_ptr, mod);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }),
      kUInt64);
}

static void sub_mod_template_(Tensor& self, const Tensor& other, uint64_t mod) {
  AT_DISPATCH_V2(
      self.scalar_type(),
      "sub_mod_cuda_",
      AT_WRAP([&]() {
        auto self_ptr = reinterpret_cast<uint64_t*>(self.data_ptr<uint64_t>());
        auto other_ptr =
            reinterpret_cast<uint64_t*>(other.data_ptr<uint64_t>());
        auto N = self.numel();
        TORCH_INTERNAL_ASSERT(
            (N == 1 << 15) || (N == 1 << 16) || (N == 1 << 17));
        auto grid = (N + block_work_size() - 1) / block_work_size();
        auto stream = at::cuda::getCurrentCUDAStream();
        fhe::sub_mod_kernel_<<<grid, block_work_size(), 0, stream>>>(
            self_ptr, other_ptr, mod);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }),
      kUInt64);
}

static void mul_mod_template(
    Tensor& c,
    const Tensor& a,
    const Tensor& b,
    uint64_t mod,
    const Tensor& barret_mu) {
  AT_DISPATCH_V2(
      a.scalar_type(),
      "mul_mod_cuda",
      AT_WRAP([&]() {
        auto a_ptr = reinterpret_cast<uint64_t*>(a.data_ptr<uint64_t>());
        auto b_ptr = reinterpret_cast<uint64_t*>(b.data_ptr<uint64_t>());
        auto c_ptr =
            reinterpret_cast<uint64_t*>(c.mutable_data_ptr<uint64_t>());
        auto mu_ptr =
            reinterpret_cast<uint64_t*>(barret_mu.data_ptr<uint64_t>());
        auto N = a.numel();
        TORCH_INTERNAL_ASSERT(
            (N == 1 << 15) || (N == 1 << 16) || (N == 1 << 17));
        auto grid = (N + block_work_size() - 1) / block_work_size();
        auto stream = at::cuda::getCurrentCUDAStream();
        fhe::mul_mod_kernel<<<grid, block_work_size(), 0, stream>>>(
            c_ptr, a_ptr, b_ptr, mod, mu_ptr);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }),
      kUInt64);
}

static void mul_mod_template_(
    Tensor& self,
    const Tensor& other,
    uint64_t mod,
    const Tensor& barret_mu) {
  AT_DISPATCH_V2(
      self.scalar_type(),
      "mul_mod_cuda_",
      AT_WRAP([&]() {
        auto self_ptr = reinterpret_cast<uint64_t*>(self.data_ptr<uint64_t>());
        auto other_ptr =
            reinterpret_cast<uint64_t*>(other.data_ptr<uint64_t>());
        auto mu_ptr =
            reinterpret_cast<uint64_t*>(barret_mu.data_ptr<uint64_t>());
        auto N = self.numel();
        TORCH_INTERNAL_ASSERT(
            (N == 1 << 15) || (N == 1 << 16) || (N == 1 << 17));
        auto grid = (N + block_work_size() - 1) / block_work_size();
        auto stream = at::cuda::getCurrentCUDAStream();
        fhe::mul_mod_kernel_<<<grid, block_work_size(), 0, stream>>>(
            self_ptr, other_ptr, mod, mu_ptr);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }),
      kUInt64);
}

Tensor add_mod_cuda(const Tensor& a, const Tensor& b, const Scalar& mod) {
  Tensor c = at::empty_like(a);
  add_mod_template(c, a, b, mod.toUInt64());
  return c;
}

Tensor& add_mod_cuda_(Tensor& self, const Tensor& other, const Scalar& mod) {
  add_mod_template_(self, other, mod.toUInt64());
  return self;
}

Tensor& add_mod_out_cuda(
    const Tensor& a,
    const Tensor& b,
    const Scalar& mod,
    Tensor& c) {
  add_mod_template(c, a, b, mod.toUInt64());
  return c;
}

Tensor sub_mod_cuda(const Tensor& a, const Tensor& b, const Scalar& mod) {
  Tensor c = at::empty_like(a);
  sub_mod_template(c, a, b, mod.toUInt64());
  return c;
}

Tensor& sub_mod_cuda_(Tensor& self, const Tensor& other, const Scalar& mod) {
  sub_mod_template_(self, other, mod.toUInt64());
  return self;
}

Tensor& sub_mod_out_cuda(
    const Tensor& a,
    const Tensor& b,
    const Scalar& mod,
    Tensor& c) {
  sub_mod_template(c, a, b, mod.toUInt64());
  return c;
}

Tensor mul_mod_cuda(
    const Tensor& a,
    const Tensor& b,
    const Scalar& mod,
    const Tensor& barret_mu) {
  TORCH_CHECK(2 == barret_mu.numel(), "The number of barret_mu should be two!");
  Tensor c = at::empty_like(a);
  mul_mod_template(c, a, b, mod.toUInt64(), barret_mu);
  return c;
}

Tensor& mul_mod_cuda_(
    Tensor& self,
    const Tensor& other,
    const Scalar& mod,
    const Tensor& barret_mu) {
  TORCH_CHECK(2 == barret_mu.numel(), "The number of barret_mu should be two!");
  mul_mod_template_(self, other, mod.toUInt64(), barret_mu);
  return self;
}

Tensor& mul_mod_out_cuda(
    const Tensor& a,
    const Tensor& b,
    const Scalar& mod,
    const Tensor& barret_mu,
    Tensor& c) {
  TORCH_CHECK(2 == barret_mu.numel(), "The number of barret_mu should be two!");
  mul_mod_template(c, a, b, mod.toUInt64(), barret_mu);
  return c;
}

} // namespace at::native