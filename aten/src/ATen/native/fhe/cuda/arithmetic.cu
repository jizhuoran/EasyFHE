#include <ATen/Dispatch_v2.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/thread_constants.h>
#include <ATen/ops/copy.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>
#include <ATen/native/fhe/cuda/arithmetic.cuh>
#include <cstdint>

#pragma clang diagnostic ignored "-Wmissing-prototypes"

#define BLOCK_SIZE (512)
#define MAX_NUM_BLOCKS (BLOCK_SIZE)

namespace fhe {

__global__ void automorphism_kernel(
    uint64_t* out,
    const uint64_t* in,
    const size_t* index) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  out[i] = in[index[i]];
}

__global__ void mod_switch_kernel(
    uint64_t* out,
    const uint64_t* in,
    uint64_t new_modulus,
    uint64_t diff,
    uint64_t old_modulus_by_two) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;

  uint64_t tmp = in[i] > old_modulus_by_two ? diff : 0;
  out[i] = add_mod(in[i], tmp, new_modulus);
}

__global__ void neg_mod_kernel(uint64_t* c, const uint64_t* a, uint64_t mod) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  c[i] = neg_mod(a[i], mod);
}

__global__ void neg_mod_kernel_(uint64_t* self, uint64_t mod) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  self[i] = neg_mod(self[i], mod);
}

__global__ void add_mod_kernel(
    const size_t N,
    const size_t L,
    uint64_t* c,
    const uint64_t* a,
    const uint64_t* b,
    const uint64_t* mod) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  auto l = blockIdx.y;
  if (i < N) {
    c[l * N + i] = add_mod(a[l * N + i], b[l * N + i], mod[l]);
  }
}

__global__ void add_mod_kernel_(
    const size_t N,
    const size_t L,
    uint64_t* self,
    const uint64_t* other,
    const uint64_t* mod) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  auto l = blockIdx.y;
  if (i < N) {
    self[l * N + i] = add_mod(self[l * N + i], other[l * N + i], mod[l]);
  }
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

__global__ void mul_scalar_mod_kernel(
    uint64_t* c,
    const uint64_t* a,
    uint64_t scalar,
    uint64_t mod,
    const uint64_t* barret_mu) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  c[i] = mul_mod(a[i], scalar, mod, barret_mu[0], barret_mu[1]);
}

__global__ void mul_scalar_mod_kernel_(
    uint64_t* self,
    uint64_t scalar,
    uint64_t mod,
    const uint64_t* barret_mu) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  self[i] = mul_mod(self[i], scalar, mod, barret_mu[0], barret_mu[1]);
}

} // namespace fhe

namespace at::native {

static void neg_mod_template(Tensor& c, const Tensor& a, uint64_t mod) {
  AT_DISPATCH_V2(
      a.scalar_type(),
      "neg_mod_cuda",
      AT_WRAP([&]() {
        auto a_ptr = reinterpret_cast<uint64_t*>(a.data_ptr<uint64_t>());
        auto c_ptr =
            reinterpret_cast<uint64_t*>(c.mutable_data_ptr<uint64_t>());
        auto N = a.numel();
        // TORCH_INTERNAL_ASSERT(
            // (N == 1 << 15) || (N == 1 << 16) || (N == 1 << 17));
        auto grid = (N + block_work_size() - 1) / block_work_size();
        auto stream = at::cuda::getCurrentCUDAStream();
        fhe::neg_mod_kernel<<<grid, block_work_size(), 0, stream>>>(
            c_ptr, a_ptr, mod);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }),
      kUInt64);
}

static void neg_mod_template_(Tensor& self, uint64_t mod) {
  AT_DISPATCH_V2(
      self.scalar_type(),
      "neg_mod_cuda_",
      AT_WRAP([&]() {
        auto self_ptr = reinterpret_cast<uint64_t*>(self.data_ptr<uint64_t>());
        auto N = self.numel();
        // TORCH_INTERNAL_ASSERT(
            // (N == 1 << 15) || (N == 1 << 16) || (N == 1 << 17));
        auto grid = (N + block_work_size() - 1) / block_work_size();
        auto stream = at::cuda::getCurrentCUDAStream();
        fhe::neg_mod_kernel_<<<grid, block_work_size(), 0, stream>>>(
            self_ptr, mod);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }),
      kUInt64);
}

static void add_mod_template(
    Tensor& c,
    const Tensor& a,
    const Tensor& b,
    const Tensor& mod,
    int64_t L) {
  TORCH_INTERNAL_ASSERT(a.dim() == 2 && b.dim() == 2);
  AT_DISPATCH_V2(
      a.scalar_type(),
      "add_mod_cuda",
      AT_WRAP([&]() {
        auto a_ptr = reinterpret_cast<uint64_t*>(a.data_ptr<uint64_t>());
        auto b_ptr = reinterpret_cast<uint64_t*>(b.data_ptr<uint64_t>());
        auto c_ptr =
            reinterpret_cast<uint64_t*>(c.mutable_data_ptr<uint64_t>());
        auto mod_ptr = reinterpret_cast<uint64_t*>(mod.data_ptr<uint64_t>());
        auto N = a.sizes()[1];
        // TORCH_INTERNAL_ASSERT(
            // (N == 1 << 15) || (N == 1 << 16) || (N == 1 << 17));
        auto grid = (N + block_work_size() - 1) / block_work_size();
        auto stream = at::cuda::getCurrentCUDAStream();
        fhe::add_mod_kernel<<<
            dim3(grid, L),
            dim3(block_work_size(), 1),
            0,
            stream>>>(N, L, c_ptr, a_ptr, b_ptr, mod_ptr);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }),
      kUInt64);
}

static void add_mod_template_(
    Tensor& self,
    const Tensor& other,
    const Tensor& mod,
    int64_t L) {
  TORCH_INTERNAL_ASSERT(self.dim() == 2 && other.dim() == 2);
  AT_DISPATCH_V2(
      self.scalar_type(),
      "add_mod_cuda_",
      AT_WRAP([&]() {
        auto self_ptr = reinterpret_cast<uint64_t*>(self.data_ptr<uint64_t>());
        auto other_ptr =
            reinterpret_cast<uint64_t*>(other.data_ptr<uint64_t>());
        auto mod_ptr = reinterpret_cast<uint64_t*>(mod.data_ptr<uint64_t>());
        auto N = self.sizes()[1];
        // TORCH_INTERNAL_ASSERT(
            // (N == 1 << 15) || (N == 1 << 16) || (N == 1 << 17));
        auto grid = (N + block_work_size() - 1) / block_work_size();
        auto stream = at::cuda::getCurrentCUDAStream();
        fhe::add_mod_kernel_<<<dim3(grid, L), block_work_size(), 0, stream>>>(
            N, L, self_ptr, other_ptr, mod_ptr);
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
        // TORCH_INTERNAL_ASSERT(
            // (N == 1 << 15) || (N == 1 << 16) || (N == 1 << 17));
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
        // TORCH_INTERNAL_ASSERT(
            // (N == 1 << 15) || (N == 1 << 16) || (N == 1 << 17));
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
        // TORCH_INTERNAL_ASSERT(
            // (N == 1 << 15) || (N == 1 << 16) || (N == 1 << 17));
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
        // TORCH_INTERNAL_ASSERT(
            // (N == 1 << 15) || (N == 1 << 16) || (N == 1 << 17));
        auto grid = (N + block_work_size() - 1) / block_work_size();
        auto stream = at::cuda::getCurrentCUDAStream();
        fhe::mul_mod_kernel_<<<grid, block_work_size(), 0, stream>>>(
            self_ptr, other_ptr, mod, mu_ptr);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }),
      kUInt64);
}

static void mul_scalar_mod_template(
    Tensor& c,
    const Tensor& a,
    uint64_t scalar,
    uint64_t mod,
    const Tensor& barret_mu) {
  AT_DISPATCH_V2(
      a.scalar_type(),
      "mul_scalar_mod_cuda",
      AT_WRAP([&]() {
        auto a_ptr = reinterpret_cast<uint64_t*>(a.data_ptr<uint64_t>());
        auto c_ptr =
            reinterpret_cast<uint64_t*>(c.mutable_data_ptr<uint64_t>());
        auto mu_ptr =
            reinterpret_cast<uint64_t*>(barret_mu.data_ptr<uint64_t>());
        auto N = a.numel();
        // TORCH_INTERNAL_ASSERT(
            // (N == 1 << 15) || (N == 1 << 16) || (N == 1 << 17));
        auto grid = (N + block_work_size() - 1) / block_work_size();
        auto stream = at::cuda::getCurrentCUDAStream();
        fhe::mul_scalar_mod_kernel<<<grid, block_work_size(), 0, stream>>>(
            c_ptr, a_ptr, scalar, mod, mu_ptr);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }),
      kUInt64);
}

static void mul_scalar_mod_template_(
    Tensor& self,
    uint64_t scalar,
    uint64_t mod,
    const Tensor& barret_mu) {
  AT_DISPATCH_V2(
      self.scalar_type(),
      "mul_scalar_mod_cuda_",
      AT_WRAP([&]() {
        auto self_ptr = reinterpret_cast<uint64_t*>(self.data_ptr<uint64_t>());
        auto mu_ptr =
            reinterpret_cast<uint64_t*>(barret_mu.data_ptr<uint64_t>());
        auto N = self.numel();
        // TORCH_INTERNAL_ASSERT(
            // (N == 1 << 15) || (N == 1 << 16) || (N == 1 << 17));
        auto grid = (N + block_work_size() - 1) / block_work_size();
        auto stream = at::cuda::getCurrentCUDAStream();
        fhe::mul_scalar_mod_kernel_<<<grid, block_work_size(), 0, stream>>>(
            self_ptr, scalar, mod, mu_ptr);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }),
      kUInt64);
}

Tensor automorphism_cuda(const Tensor& input, const Tensor& index) {
  Tensor output = at::empty_like(input);
  AT_DISPATCH_V2(
      input.scalar_type(),
      "automorphism_cuda",
      AT_WRAP([&]() {
        auto in_ptr = reinterpret_cast<uint64_t*>(input.data_ptr<uint64_t>());
        auto index_ptr =
            reinterpret_cast<uint64_t*>(index.data_ptr<uint64_t>());
        auto out_ptr =
            reinterpret_cast<uint64_t*>(output.mutable_data_ptr<uint64_t>());
        auto N = input.numel();
        // TORCH_INTERNAL_ASSERT(
            // (N == 1 << 15) || (N == 1 << 16) || (N == 1 << 17));
        auto grid = (N + block_work_size() - 1) / block_work_size();
        auto stream = at::cuda::getCurrentCUDAStream();
        fhe::automorphism_kernel<<<grid, block_work_size(), 0, stream>>>(
            out_ptr, in_ptr, index_ptr);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }),
      kUInt64);
  return output;
}

Tensor mod_switch_cuda(
    const Tensor& input,
    const Scalar& new_modulus,
    const Scalar& old_modulus) {
  uint64_t new_modulus_ = new_modulus.toUInt64();
  uint64_t old_modulus_ = old_modulus.toUInt64();
  uint64_t old_modulus_by_two = old_modulus_ >> 1;
  uint64_t diff = new_modulus_ - old_modulus_;

  Tensor output = at::empty_like(input);
  AT_DISPATCH_V2(
      input.scalar_type(),
      "mod_switch_cuda",
      AT_WRAP([&]() {
        auto in_ptr = reinterpret_cast<uint64_t*>(input.data_ptr<uint64_t>());
        auto out_ptr =
            reinterpret_cast<uint64_t*>(output.mutable_data_ptr<uint64_t>());
        auto N = input.numel();
        // TORCH_INTERNAL_ASSERT(
            // (N == 1 << 15) || (N == 1 << 16) || (N == 1 << 17));
        auto grid = (N + block_work_size() - 1) / block_work_size();
        auto stream = at::cuda::getCurrentCUDAStream();
        fhe::mod_switch_kernel<<<grid, block_work_size(), 0, stream>>>(
            out_ptr, in_ptr, new_modulus_, diff, old_modulus_by_two);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }),
      kUInt64);
  return output;
}

Tensor neg_mod_cuda(const Tensor& a, const Scalar& mod) {
  Tensor c = at::empty_like(a);
  neg_mod_template(c, a, mod.toUInt64());
  return c;
}

Tensor& neg_mod_cuda_(Tensor& self, const Scalar& mod) {
  neg_mod_template_(self, mod.toUInt64());
  return self;
}

Tensor& neg_mod_out_cuda(const Tensor& a, const Scalar& mod, Tensor& c) {
  neg_mod_template(c, a, mod.toUInt64());
  return c;
}

Tensor add_mod_cuda(
    const Tensor& a,
    const Tensor& b,
    const Tensor& mod,
    int64_t L) {
  Tensor c = at::empty_like(a);
  add_mod_template(c, a, b, mod, L);
  return c;
}

Tensor& add_mod_cuda_(
    Tensor& self,
    const Tensor& other,
    const Tensor& mod,
    int64_t L) {
  add_mod_template_(self, other, mod, L);
  return self;
}

Tensor& add_mod_out_cuda(
    const Tensor& a,
    const Tensor& b,
    const Tensor& mod,
    int64_t L,
    Tensor& c) {
  add_mod_template(c, a, b, mod, L);
  return c;
}

Tensor add_scalar_mod_cuda(
    const Tensor& a,
    const Scalar& scalar,
    const Scalar& mod) {
  Tensor c = at::empty_like(a);
  // add_scalar_mod_template(c, a, scalar.toUInt64(), mod.toUInt64());
  return c;
}

Tensor& add_scalar_mod_cuda_(
    Tensor& self,
    const Scalar& scalar,
    const Scalar& mod) {
  // add_scalar_mod_template_(self, scalar.toUInt64(), mod.toUInt64());
  return self;
}

Tensor& add_scalar_mod_out_cuda(
    const Tensor& a,
    const Scalar& b,
    const Scalar& mod,
    Tensor& c) {
  // add_scalar_mod_template(c, a, b.toUInt64(), mod.toUInt64());
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

Tensor mul_scalar_mod_cuda(
    const Tensor& a,
    const Scalar& scalar,
    const Scalar& mod,
    const Tensor& barret_mu) {
  TORCH_CHECK(2 == barret_mu.numel(), "The number of barret_mu should be two!");
  Tensor c = at::empty_like(a);
  mul_scalar_mod_template(c, a, scalar.toUInt64(), mod.toUInt64(), barret_mu);
  return c;
}

Tensor& mul_scalar_mod_cuda_(
    Tensor& self,
    const Scalar& scalar,
    const Scalar& mod,
    const Tensor& barret_mu) {
  TORCH_CHECK(2 == barret_mu.numel(), "The number of barret_mu should be two!");
  mul_scalar_mod_template_(self, scalar.toUInt64(), mod.toUInt64(), barret_mu);
  return self;
}

Tensor& mul_scalar_mod_out_cuda(
    const Tensor& a,
    const Scalar& scalar,
    const Scalar& mod,
    const Tensor& barret_mu,
    Tensor& c) {
  TORCH_CHECK(2 == barret_mu.numel(), "The number of barret_mu should be two!");
  mul_scalar_mod_template(c, a, scalar.toUInt64(), mod.toUInt64(), barret_mu);
  return c;
}

} // namespace at::native