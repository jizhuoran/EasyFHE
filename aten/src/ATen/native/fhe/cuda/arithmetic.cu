// #include <ATen/Dispatch.h>
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

#pragma clang diagnostic ignored "-Wmissing-prototypes"

#define BLOCK_SIZE (512)
#define MAX_NUM_BLOCKS (BLOCK_SIZE)

namespace fhe {

__device__ __forceinline__ uint64_t
add_mod(uint64_t a, uint64_t b, uint64_t mod) {
  uint64_t res = a + b;
  return res >= mod ? res - mod : res;
}

__global__ void add_mod_kernel(
    const int64_t N,
    uint64_t* c,
    const uint64_t* a,
    const uint64_t* b,
    uint64_t mod) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    c[i] = add_mod(a[i], b[i], mod);
  }
}

__global__ void add_mod_kernel_(
    const int64_t N,
    uint64_t* self,
    const uint64_t* other,
    uint64_t mod) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    self[i] = add_mod(self[i], other[i], mod);
  }
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
            N > 0 && N <= std::numeric_limits<int32_t>::max());
        auto grid = (N + block_work_size() - 1) / block_work_size();
        auto stream = at::cuda::getCurrentCUDAStream();
        fhe::add_mod_kernel<<<grid, block_work_size(), 0, stream>>>(
            N, c_ptr, a_ptr, b_ptr, mod);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }),
      kUInt64);
}

Tensor add_mod_cuda(const Tensor& a, const Tensor& b, int64_t mod) {
  Tensor c = at::empty_like(a);
  add_mod_template(c, a, b, mod);
  return c;
}

Tensor& add_mod_cuda_(Tensor& self, const Tensor& other, int64_t mod) {
  add_mod_template(self, self, other, mod);
  return self;
}

Tensor& add_mod_out_cuda(
    const Tensor& a,
    const Tensor& b,
    int64_t mod,
    Tensor& c) {
  add_mod_template(c, a, b, mod);
  return c;
}

} // namespace at::native