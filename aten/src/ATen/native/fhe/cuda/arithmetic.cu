#include <ATen/Dispatch_v2.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/thread_constants.h>
#include <ATen/native/fhe/cuda/arithmetic.h>
#include <ATen/ops/empty.h>
#include <ATen/native/fhe/cuda/Utils.cuh>

#pragma clang diagnostic ignored "-Wmissing-prototypes"

namespace {

using at::Tensor;

enum class ArithmeticOp {
  Add,
  Sub,
  Mul,
  Neg,
};

enum class RhsLayout {
  Tensor,
  ScalarByLimb,
};

template <RhsLayout Layout>
__device__ __forceinline__ uint64_t rhs_value(
    const uint64_t* __restrict__ rhs,
    size_t cv_idx,
    size_t batch_idx,
    size_t limb_idx,
    size_t coeff_idx,
    size_t N,
    size_t LN_B,
    size_t BLN_B) {
  if constexpr (Layout == RhsLayout::ScalarByLimb) {
    return rhs[limb_idx];
  } else {
    return rhs[cv_idx * BLN_B + batch_idx * LN_B + limb_idx * N + coeff_idx];
  }
}

template <ArithmeticOp Op, bool HasBarrett>
__device__ __forceinline__ uint64_t apply_op(
    uint64_t lhs,
    uint64_t rhs,
    uint64_t mod,
    const uint64_t* __restrict__ barrett_mu,
    size_t limb_idx) {
  if constexpr (Op == ArithmeticOp::Add) {
    return fhe::add_mod(lhs, rhs, mod);
  } else if constexpr (Op == ArithmeticOp::Sub) {
    return fhe::sub_mod(lhs, rhs, mod);
  } else if constexpr (Op == ArithmeticOp::Mul) {
    static_assert(HasBarrett, "mul_mod requires Barrett parameters");
    return fhe::mul_mod(
        lhs, rhs, mod, barrett_mu[limb_idx * 2], barrett_mu[limb_idx * 2 + 1]);
  } else {
    return fhe::neg_mod(lhs, 0, mod);
  }
}

template <ArithmeticOp Op, RhsLayout Layout, bool HasBarrett>
__global__ void arithmetic_kernel(
    size_t N,
    size_t batch,
    size_t LN_C,
    size_t LN_A,
    size_t LN_B,
    size_t BLN_C,
    size_t BLN_A,
    size_t BLN_B,
    uint64_t* __restrict__ out,
    const uint64_t* __restrict__ lhs,
    const uint64_t* __restrict__ rhs,
    const uint64_t* __restrict__ mod,
    const uint64_t* __restrict__ barrett_mu) {
  const size_t coeff_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (coeff_idx >= N) {
    return;
  }

  const size_t limb_idx = blockIdx.y;
  const size_t cv_idx = blockIdx.z / batch;
  const size_t batch_idx = blockIdx.z - cv_idx * batch;
  const uint64_t modulus = mod[limb_idx];

  const size_t lhs_offset =
      cv_idx * BLN_A + batch_idx * LN_A + limb_idx * N + coeff_idx;
  const size_t out_offset =
      cv_idx * BLN_C + batch_idx * LN_C + limb_idx * N + coeff_idx;
  uint64_t rhs_element = 0;
  if constexpr (Op != ArithmeticOp::Neg) {
    rhs_element = rhs_value<Layout>(
        rhs, cv_idx, batch_idx, limb_idx, coeff_idx, N, LN_B, BLN_B);
  }
  out[out_offset] = apply_op<Op, HasBarrett>(
      lhs[lhs_offset], rhs_element, modulus, barrett_mu, limb_idx);
}

template <ArithmeticOp Op, RhsLayout Layout, bool HasBarrett>
void launch_arithmetic_kernel(
    size_t num_cv,
    size_t batch,
    size_t L_C,
    size_t L_A,
    size_t L_B,
    size_t N,
    int64_t cur_limbs,
    uint64_t* out,
    const uint64_t* lhs,
    const uint64_t* rhs,
    const uint64_t* mod,
    const uint64_t* barrett_mu) {
  const size_t LN_C = L_C * N;
  const size_t LN_A = L_A * N;
  const size_t LN_B = L_B * N;
  const size_t BLN_C = batch * LN_C;
  const size_t BLN_A = batch * LN_A;
  const size_t BLN_B = batch * LN_B;
  const dim3 grid(num_blocks(N), cur_limbs, batch * num_cv);
  const dim3 block(BLOCK_SIZE);
  const auto stream = at::cuda::getCurrentCUDAStream();

  arithmetic_kernel<Op, Layout, HasBarrett><<<grid, block, 0, stream>>>(
      N,
      batch,
      LN_C,
      LN_A,
      LN_B,
      BLN_C,
      BLN_A,
      BLN_B,
      out,
      lhs,
      rhs,
      mod,
      barrett_mu);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <ArithmeticOp Op, RhsLayout Layout, bool HasBarrett>
void run_arithmetic(
    Tensor& out,
    const Tensor& lhs,
    const Tensor& rhs,
    const Tensor& mod,
    const Tensor* barrett_mu,
    int64_t cur_limbs) {
  TORCH_INTERNAL_ASSERT(lhs.dim() == 4);
  const auto num_cv = lhs.sizes()[0];
  const auto batch = lhs.sizes()[1];
  const auto N = lhs.sizes()[3];
  TORCH_INTERNAL_ASSERT(
      (N == 1 << 6) || (N == 1 << 14) || (N == 1 << 15) || (N == 1 << 16) ||
      (N == 1 << 17) || (N == 1 << 18));
  if constexpr (HasBarrett) {
    TORCH_INTERNAL_ASSERT(barrett_mu != nullptr);
  }
  const auto rhs_limb_extent =
      Layout == RhsLayout::Tensor ? rhs.sizes()[2] : cur_limbs;

  launch_arithmetic_kernel<Op, Layout, HasBarrett>(
      num_cv,
      batch,
      out.sizes()[2],
      lhs.sizes()[2],
      rhs_limb_extent,
      N,
      cur_limbs,
      out.mutable_data_ptr<uint64_t>(),
      lhs.data_ptr<uint64_t>(),
      rhs.data_ptr<uint64_t>(),
      mod.data_ptr<uint64_t>(),
      HasBarrett ? barrett_mu->data_ptr<uint64_t>() : nullptr);
}

template <ArithmeticOp Op, RhsLayout Layout, bool HasBarrett>
Tensor make_arithmetic_result(
    const Tensor& lhs,
    const Tensor& rhs,
    const Tensor& mod,
    const Tensor* barrett_mu,
    int64_t cur_limbs) {
  Tensor out = at::empty(
      {lhs.sizes()[0], lhs.sizes()[1], cur_limbs, lhs.sizes()[3]},
      lhs.options());
  run_arithmetic<Op, Layout, HasBarrett>(
      out, lhs, rhs, mod, barrett_mu, cur_limbs);
  return out;
}

template <ArithmeticOp Op, RhsLayout Layout, bool HasBarrett>
Tensor& arithmetic_inplace(
    Tensor& self,
    const Tensor& rhs,
    const Tensor& mod,
    const Tensor* barrett_mu,
    int64_t cur_limbs) {
  run_arithmetic<Op, Layout, HasBarrett>(
      self, self, rhs, mod, barrett_mu, cur_limbs);
  return self;
}

template <ArithmeticOp Op, RhsLayout Layout, bool HasBarrett>
Tensor& arithmetic_out(
    const Tensor& lhs,
    const Tensor& rhs,
    const Tensor& mod,
    const Tensor* barrett_mu,
    int64_t cur_limbs,
    Tensor& out) {
  run_arithmetic<Op, Layout, HasBarrett>(
      out, lhs, rhs, mod, barrett_mu, cur_limbs);
  return out;
}

} // namespace

namespace at::native {

void vadd_mod(
    const size_t num_cv,
    const size_t batch,
    const size_t L_C,
    const size_t L_A,
    const size_t L_B,
    const size_t N,
    int64_t cur_limbs,
    uint64_t* out,
    const uint64_t* lhs,
    const uint64_t* rhs,
    const uint64_t* mod) {
  launch_arithmetic_kernel<ArithmeticOp::Add, RhsLayout::Tensor, false>(
      num_cv, batch, L_C, L_A, L_B, N, cur_limbs, out, lhs, rhs, mod, nullptr);
}

void vsub_mod(
    const size_t num_cv,
    const size_t batch,
    const size_t L_C,
    const size_t L_A,
    const size_t L_B,
    const size_t N,
    int64_t cur_limbs,
    uint64_t* out,
    const uint64_t* lhs,
    const uint64_t* rhs,
    const uint64_t* mod) {
  launch_arithmetic_kernel<ArithmeticOp::Sub, RhsLayout::Tensor, false>(
      num_cv, batch, L_C, L_A, L_B, N, cur_limbs, out, lhs, rhs, mod, nullptr);
}

void vmul_mod(
    const size_t num_cv,
    const size_t batch,
    const size_t L_C,
    const size_t L_A,
    const size_t L_B,
    const size_t N,
    int64_t cur_limbs,
    uint64_t* out,
    const uint64_t* lhs,
    const uint64_t* rhs,
    const uint64_t* mod,
    const uint64_t* barrett_mu) {
  launch_arithmetic_kernel<ArithmeticOp::Mul, RhsLayout::Tensor, true>(
      num_cv,
      batch,
      L_C,
      L_A,
      L_B,
      N,
      cur_limbs,
      out,
      lhs,
      rhs,
      mod,
      barrett_mu);
}

void vneg_mod(
    const size_t num_cv,
    const size_t batch,
    const size_t L_C,
    const size_t L_A,
    const size_t L_B,
    const size_t N,
    int64_t cur_limbs,
    uint64_t* out,
    const uint64_t* lhs,
    const uint64_t* rhs,
    const uint64_t* mod) {
  launch_arithmetic_kernel<ArithmeticOp::Neg, RhsLayout::Tensor, false>(
      num_cv, batch, L_C, L_A, L_B, N, cur_limbs, out, lhs, rhs, mod, nullptr);
}

Tensor add_mod_cuda(
    const Tensor& lhs,
    const Tensor& rhs,
    const Tensor& mod,
    int64_t cur_limbs) {
  return make_arithmetic_result<ArithmeticOp::Add, RhsLayout::Tensor, false>(
      lhs, rhs, mod, nullptr, cur_limbs);
}

Tensor& add_mod_cuda_(
    Tensor& self,
    const Tensor& rhs,
    const Tensor& mod,
    int64_t cur_limbs) {
  return arithmetic_inplace<ArithmeticOp::Add, RhsLayout::Tensor, false>(
      self, rhs, mod, nullptr, cur_limbs);
}

Tensor& add_mod_out_cuda(
    const Tensor& lhs,
    const Tensor& rhs,
    const Tensor& mod,
    int64_t cur_limbs,
    Tensor& out) {
  return arithmetic_out<ArithmeticOp::Add, RhsLayout::Tensor, false>(
      lhs, rhs, mod, nullptr, cur_limbs, out);
}

Tensor sub_mod_cuda(
    const Tensor& lhs,
    const Tensor& rhs,
    const Tensor& mod,
    int64_t cur_limbs) {
  return make_arithmetic_result<ArithmeticOp::Sub, RhsLayout::Tensor, false>(
      lhs, rhs, mod, nullptr, cur_limbs);
}

Tensor& sub_mod_cuda_(
    Tensor& self,
    const Tensor& rhs,
    const Tensor& mod,
    int64_t cur_limbs) {
  return arithmetic_inplace<ArithmeticOp::Sub, RhsLayout::Tensor, false>(
      self, rhs, mod, nullptr, cur_limbs);
}

Tensor& sub_mod_out_cuda(
    const Tensor& lhs,
    const Tensor& rhs,
    const Tensor& mod,
    int64_t cur_limbs,
    Tensor& out) {
  return arithmetic_out<ArithmeticOp::Sub, RhsLayout::Tensor, false>(
      lhs, rhs, mod, nullptr, cur_limbs, out);
}

Tensor mul_mod_cuda(
    const Tensor& lhs,
    const Tensor& rhs,
    const Tensor& mod,
    const Tensor& barrett_mu,
    int64_t cur_limbs) {
  return make_arithmetic_result<ArithmeticOp::Mul, RhsLayout::Tensor, true>(
      lhs, rhs, mod, &barrett_mu, cur_limbs);
}

Tensor& mul_mod_cuda_(
    Tensor& self,
    const Tensor& rhs,
    const Tensor& mod,
    const Tensor& barrett_mu,
    int64_t cur_limbs) {
  return arithmetic_inplace<ArithmeticOp::Mul, RhsLayout::Tensor, true>(
      self, rhs, mod, &barrett_mu, cur_limbs);
}

Tensor& mul_mod_out_cuda(
    const Tensor& lhs,
    const Tensor& rhs,
    const Tensor& mod,
    const Tensor& barrett_mu,
    int64_t cur_limbs,
    Tensor& out) {
  return arithmetic_out<ArithmeticOp::Mul, RhsLayout::Tensor, true>(
      lhs, rhs, mod, &barrett_mu, cur_limbs, out);
}

Tensor add_scalar_mod_cuda(
    const Tensor& lhs,
    const Tensor& rhs,
    const Tensor& mod,
    int64_t cur_limbs) {
  return make_arithmetic_result<
      ArithmeticOp::Add,
      RhsLayout::ScalarByLimb,
      false>(lhs, rhs, mod, nullptr, cur_limbs);
}

Tensor& add_scalar_mod_cuda_(
    Tensor& self,
    const Tensor& rhs,
    const Tensor& mod,
    int64_t cur_limbs) {
  return arithmetic_inplace<ArithmeticOp::Add, RhsLayout::ScalarByLimb, false>(
      self, rhs, mod, nullptr, cur_limbs);
}

Tensor& add_scalar_mod_out_cuda(
    const Tensor& lhs,
    const Tensor& rhs,
    const Tensor& mod,
    int64_t cur_limbs,
    Tensor& out) {
  return arithmetic_out<ArithmeticOp::Add, RhsLayout::ScalarByLimb, false>(
      lhs, rhs, mod, nullptr, cur_limbs, out);
}

Tensor sub_scalar_mod_cuda(
    const Tensor& lhs,
    const Tensor& rhs,
    const Tensor& mod,
    int64_t cur_limbs) {
  return make_arithmetic_result<
      ArithmeticOp::Sub,
      RhsLayout::ScalarByLimb,
      false>(lhs, rhs, mod, nullptr, cur_limbs);
}

Tensor& sub_scalar_mod_cuda_(
    Tensor& self,
    const Tensor& rhs,
    const Tensor& mod,
    int64_t cur_limbs) {
  return arithmetic_inplace<ArithmeticOp::Sub, RhsLayout::ScalarByLimb, false>(
      self, rhs, mod, nullptr, cur_limbs);
}

Tensor& sub_scalar_mod_out_cuda(
    const Tensor& lhs,
    const Tensor& rhs,
    const Tensor& mod,
    int64_t cur_limbs,
    Tensor& out) {
  return arithmetic_out<ArithmeticOp::Sub, RhsLayout::ScalarByLimb, false>(
      lhs, rhs, mod, nullptr, cur_limbs, out);
}

Tensor mul_scalar_mod_cuda(
    const Tensor& lhs,
    const Tensor& rhs,
    const Tensor& mod,
    const Tensor& barrett_mu,
    int64_t cur_limbs) {
  return make_arithmetic_result<
      ArithmeticOp::Mul,
      RhsLayout::ScalarByLimb,
      true>(lhs, rhs, mod, &barrett_mu, cur_limbs);
}

Tensor& mul_scalar_mod_cuda_(
    Tensor& self,
    const Tensor& rhs,
    const Tensor& mod,
    const Tensor& barrett_mu,
    int64_t cur_limbs) {
  return arithmetic_inplace<ArithmeticOp::Mul, RhsLayout::ScalarByLimb, true>(
      self, rhs, mod, &barrett_mu, cur_limbs);
}

Tensor& mul_scalar_mod_out_cuda(
    const Tensor& lhs,
    const Tensor& rhs,
    const Tensor& mod,
    const Tensor& barrett_mu,
    int64_t cur_limbs,
    Tensor& out) {
  return arithmetic_out<ArithmeticOp::Mul, RhsLayout::ScalarByLimb, true>(
      lhs, rhs, mod, &barrett_mu, cur_limbs, out);
}

Tensor neg_mod_cuda(
    const Tensor& lhs,
    const Tensor& rhs,
    const Tensor& mod,
    int64_t cur_limbs) {
  return make_arithmetic_result<ArithmeticOp::Neg, RhsLayout::Tensor, false>(
      lhs, rhs, mod, nullptr, cur_limbs);
}

Tensor& neg_mod_cuda_(
    Tensor& self,
    const Tensor& rhs,
    const Tensor& mod,
    int64_t cur_limbs) {
  return arithmetic_inplace<ArithmeticOp::Neg, RhsLayout::Tensor, false>(
      self, rhs, mod, nullptr, cur_limbs);
}

Tensor& neg_mod_out_cuda(
    const Tensor& lhs,
    const Tensor& rhs,
    const Tensor& mod,
    int64_t cur_limbs,
    Tensor& out) {
  return arithmetic_out<ArithmeticOp::Neg, RhsLayout::Tensor, false>(
      lhs, rhs, mod, nullptr, cur_limbs, out);
}

} // namespace at::native
