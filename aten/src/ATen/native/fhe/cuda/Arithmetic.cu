#include <ATen/Dispatch_v2.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/thread_constants.h>
#include <ATen/ops/empty.h>
#include <ATen/native/fhe/cuda/device/Launch.cuh>
#include <ATen/native/fhe/cuda/device/Modular.cuh>
#include <vector>

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

enum class FusedRhsLayout {
  TensorPair,
  Plaintext,
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
  TORCH_INTERNAL_ASSERT(lhs.dim() == 3);
  const auto num_cv = 1;
  const auto batch = lhs.sizes()[0];
  const auto N = lhs.sizes()[2];
  TORCH_INTERNAL_ASSERT(
      (N == 1 << 6) || (N == 1 << 14) || (N == 1 << 15) || (N == 1 << 16) ||
      (N == 1 << 17) || (N == 1 << 18));
  if constexpr (HasBarrett) {
    TORCH_INTERNAL_ASSERT(barrett_mu != nullptr);
  }
  const auto rhs_limb_extent =
      Layout == RhsLayout::Tensor ? rhs.sizes()[1] : cur_limbs;

  launch_arithmetic_kernel<Op, Layout, HasBarrett>(
      num_cv,
      batch,
      out.sizes()[1],
      lhs.sizes()[1],
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
      {lhs.sizes()[0], lhs.sizes()[1], lhs.sizes()[2]},
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

inline int64_t active_limb_dim(const Tensor& tensor) {
  TORCH_INTERNAL_ASSERT(tensor.dim() == 3);
  return tensor.sizes()[1];
}

inline int64_t coeff_dim(const Tensor& tensor) {
  TORCH_INTERNAL_ASSERT(tensor.dim() == 3);
  return tensor.sizes()[2];
}

inline int64_t batch_dim(const Tensor& tensor) {
  TORCH_INTERNAL_ASSERT(tensor.dim() == 3);
  return tensor.sizes()[0];
}

inline std::vector<int64_t> fused_output_sizes(const Tensor& base, int64_t cur_limbs) {
  auto sizes = base.sizes().vec();
  sizes[base.dim() - 2] = cur_limbs;
  return sizes;
}

template <ArithmeticOp Op, FusedRhsLayout Layout, bool HasBarrett>
__global__ void fused_pair_arithmetic_kernel(
    size_t N,
    size_t batch,
    size_t L_OUT,
    size_t L_A,
    size_t L_R,
    int64_t cur_limbs,
    uint64_t* __restrict__ out0,
    uint64_t* __restrict__ out1,
    const uint64_t* __restrict__ a0,
    const uint64_t* __restrict__ a1,
    const uint64_t* __restrict__ rhs0,
    const uint64_t* __restrict__ rhs1,
    size_t rhs_batch,
    const uint64_t* __restrict__ mod,
    const uint64_t* __restrict__ barrett_mu) {
  const size_t linear = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = batch * static_cast<size_t>(cur_limbs) * N;
  if (linear >= total) {
    return;
  }

  const size_t coeff_idx = linear % N;
  const size_t limb_idx = (linear / N) % static_cast<size_t>(cur_limbs);
  const size_t batch_idx = linear / (N * static_cast<size_t>(cur_limbs));
  const uint64_t modulus = mod[limb_idx];

  const size_t out_offset = batch_idx * L_OUT * N + limb_idx * N + coeff_idx;
  const size_t lhs_offset = batch_idx * L_A * N + limb_idx * N + coeff_idx;

  uint64_t rhs_value0 = 0;
  uint64_t rhs_value1 = 0;
  if constexpr (Layout == FusedRhsLayout::TensorPair) {
    const size_t rhs_offset = batch_idx * L_R * N + limb_idx * N + coeff_idx;
    rhs_value0 = rhs0[rhs_offset];
    rhs_value1 = rhs1[rhs_offset];
  } else if constexpr (Layout == FusedRhsLayout::Plaintext) {
    const size_t rhs_batch_idx = rhs_batch == 1 ? 0 : batch_idx;
    const size_t rhs_offset = rhs_batch_idx * L_R * N + limb_idx * N + coeff_idx;
    rhs_value0 = rhs0[rhs_offset];
    rhs_value1 = rhs_value0;
  } else {
    rhs_value0 = rhs0[limb_idx];
    rhs_value1 = rhs_value0;
  }

  out0[out_offset] = apply_op<Op, HasBarrett>(
      a0[lhs_offset], rhs_value0, modulus, barrett_mu, limb_idx);
  out1[out_offset] = apply_op<Op, HasBarrett>(
      a1[lhs_offset], rhs_value1, modulus, barrett_mu, limb_idx);
}

template <ArithmeticOp Op, FusedRhsLayout Layout, bool HasBarrett>
std::vector<Tensor> fused_pair_arithmetic_out(
    const Tensor& out0,
    const Tensor& out1,
    const Tensor& a0,
    const Tensor& a1,
    const Tensor& rhs0,
    const Tensor* rhs1,
    const Tensor& mod,
    const Tensor* barrett_mu,
    int64_t cur_limbs) {
  TORCH_INTERNAL_ASSERT(a0.dim() == 3);
  TORCH_INTERNAL_ASSERT(a1.sizes() == a0.sizes());
  TORCH_INTERNAL_ASSERT(cur_limbs >= 0);
  TORCH_INTERNAL_ASSERT(cur_limbs <= active_limb_dim(a0));
  TORCH_INTERNAL_ASSERT(coeff_dim(a0) == coeff_dim(a1));
  if constexpr (Layout == FusedRhsLayout::TensorPair) {
    TORCH_INTERNAL_ASSERT(rhs1 != nullptr);
    TORCH_INTERNAL_ASSERT(rhs0.dim() == 3);
    TORCH_INTERNAL_ASSERT(rhs1->dim() == 3);
    TORCH_INTERNAL_ASSERT(batch_dim(rhs0) == batch_dim(a0));
    TORCH_INTERNAL_ASSERT(batch_dim(*rhs1) == batch_dim(a0));
    TORCH_INTERNAL_ASSERT(coeff_dim(rhs0) == coeff_dim(a0));
    TORCH_INTERNAL_ASSERT(coeff_dim(*rhs1) == coeff_dim(a0));
    TORCH_INTERNAL_ASSERT(cur_limbs <= active_limb_dim(rhs0));
    TORCH_INTERNAL_ASSERT(cur_limbs <= active_limb_dim(*rhs1));
  } else if constexpr (Layout == FusedRhsLayout::Plaintext) {
    TORCH_INTERNAL_ASSERT(rhs0.dim() == 3);
    TORCH_INTERNAL_ASSERT(coeff_dim(rhs0) == coeff_dim(a0));
    TORCH_INTERNAL_ASSERT(cur_limbs <= active_limb_dim(rhs0));
    const auto rhs_batch = batch_dim(rhs0);
    TORCH_INTERNAL_ASSERT(rhs_batch == 1 || rhs_batch == batch_dim(a0));
  } else {
    TORCH_INTERNAL_ASSERT(rhs0.dim() == 1);
    TORCH_INTERNAL_ASSERT(cur_limbs <= rhs0.sizes()[0]);
  }
  if constexpr (HasBarrett) {
    TORCH_INTERNAL_ASSERT(barrett_mu != nullptr);
  }
  TORCH_INTERNAL_ASSERT(out0.dim() == 3);
  TORCH_INTERNAL_ASSERT(out1.dim() == 3);
  TORCH_INTERNAL_ASSERT(batch_dim(out0) == batch_dim(a0));
  TORCH_INTERNAL_ASSERT(batch_dim(out1) == batch_dim(a0));
  TORCH_INTERNAL_ASSERT(coeff_dim(out0) == coeff_dim(a0));
  TORCH_INTERNAL_ASSERT(coeff_dim(out1) == coeff_dim(a0));
  TORCH_INTERNAL_ASSERT(cur_limbs <= active_limb_dim(out0));
  TORCH_INTERNAL_ASSERT(cur_limbs <= active_limb_dim(out1));
  TORCH_CHECK(out0.is_contiguous(), "fused output c0 must be contiguous");
  TORCH_CHECK(out1.is_contiguous(), "fused output c1 must be contiguous");

  const auto N = static_cast<size_t>(coeff_dim(a0));
  const auto batch = static_cast<size_t>(batch_dim(a0));
  const auto L_OUT = static_cast<size_t>(active_limb_dim(out0));
  const auto L_A = static_cast<size_t>(active_limb_dim(a0));
  const auto L_R = Layout == FusedRhsLayout::ScalarByLimb ? 0 : static_cast<size_t>(active_limb_dim(rhs0));
  const auto rhs_batch = Layout == FusedRhsLayout::Plaintext ? static_cast<size_t>(batch_dim(rhs0)) : batch;
  const size_t total = batch * static_cast<size_t>(cur_limbs) * N;
  const dim3 grid(fhe::launch_blocks(total));
  const dim3 block(fhe::kBlockSize);
  const auto stream = at::cuda::getCurrentCUDAStream();

  fused_pair_arithmetic_kernel<Op, Layout, HasBarrett><<<grid, block, 0, stream>>>(
      N,
      batch,
      L_OUT,
      L_A,
      L_R,
      cur_limbs,
      out0.mutable_data_ptr<uint64_t>(),
      out1.mutable_data_ptr<uint64_t>(),
      a0.data_ptr<uint64_t>(),
      a1.data_ptr<uint64_t>(),
      rhs0.data_ptr<uint64_t>(),
      rhs1 == nullptr ? nullptr : rhs1->data_ptr<uint64_t>(),
      rhs_batch,
      mod.data_ptr<uint64_t>(),
      HasBarrett ? barrett_mu->data_ptr<uint64_t>() : nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {out0, out1};
}

template <ArithmeticOp Op, FusedRhsLayout Layout, bool HasBarrett>
std::vector<Tensor> fused_pair_arithmetic(
    const Tensor& a0,
    const Tensor& a1,
    const Tensor& rhs0,
    const Tensor* rhs1,
    const Tensor& mod,
    const Tensor* barrett_mu,
    int64_t cur_limbs) {
  auto out0 = at::empty(fused_output_sizes(a0, cur_limbs), a0.options());
  auto out1 = at::empty(fused_output_sizes(a1, cur_limbs), a1.options());
  return fused_pair_arithmetic_out<Op, Layout, HasBarrett>(
      out0, out1, a0, a1, rhs0, rhs1, mod, barrett_mu, cur_limbs);
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

std::vector<Tensor> cv_add_pair_cuda(
    const Tensor& in0_c0,
    const Tensor& in0_c1,
    const Tensor& in1_c0,
    const Tensor& in1_c1,
    const Tensor& mod,
    int64_t cur_limbs) {
  return fused_pair_arithmetic<ArithmeticOp::Add, FusedRhsLayout::TensorPair, false>(
      in0_c0, in0_c1, in1_c0, &in1_c1, mod, nullptr, cur_limbs);
}

Tensor& cv_add_pair_inplace_cuda(
    Tensor& in0_c0,
    Tensor& in0_c1,
    const Tensor& in1_c0,
    const Tensor& in1_c1,
    const Tensor& mod,
    int64_t cur_limbs) {
  fused_pair_arithmetic_out<ArithmeticOp::Add, FusedRhsLayout::TensorPair, false>(
      in0_c0, in0_c1, in0_c0, in0_c1, in1_c0, &in1_c1, mod, nullptr, cur_limbs);
  return in0_c0;
}

std::vector<Tensor> cv_sub_pair_cuda(
    const Tensor& in0_c0,
    const Tensor& in0_c1,
    const Tensor& in1_c0,
    const Tensor& in1_c1,
    const Tensor& mod,
    int64_t cur_limbs) {
  return fused_pair_arithmetic<ArithmeticOp::Sub, FusedRhsLayout::TensorPair, false>(
      in0_c0, in0_c1, in1_c0, &in1_c1, mod, nullptr, cur_limbs);
}

Tensor& cv_sub_pair_inplace_cuda(
    Tensor& in0_c0,
    Tensor& in0_c1,
    const Tensor& in1_c0,
    const Tensor& in1_c1,
    const Tensor& mod,
    int64_t cur_limbs) {
  fused_pair_arithmetic_out<ArithmeticOp::Sub, FusedRhsLayout::TensorPair, false>(
      in0_c0, in0_c1, in0_c0, in0_c1, in1_c0, &in1_c1, mod, nullptr, cur_limbs);
  return in0_c0;
}

std::vector<Tensor> cv_mul_pt_pair_cuda(
    const Tensor& c0,
    const Tensor& c1,
    const Tensor& plaintext,
    const Tensor& mod,
    const Tensor& barrett_mu,
    int64_t cur_limbs) {
  return fused_pair_arithmetic<ArithmeticOp::Mul, FusedRhsLayout::Plaintext, true>(
      c0, c1, plaintext, nullptr, mod, &barrett_mu, cur_limbs);
}

Tensor& cv_mul_pt_pair_inplace_cuda(
    Tensor& c0,
    Tensor& c1,
    const Tensor& plaintext,
    const Tensor& mod,
    const Tensor& barrett_mu,
    int64_t cur_limbs) {
  fused_pair_arithmetic_out<ArithmeticOp::Mul, FusedRhsLayout::Plaintext, true>(
      c0, c1, c0, c1, plaintext, nullptr, mod, &barrett_mu, cur_limbs);
  return c0;
}

std::vector<Tensor> cv_mul_scalar_pair_cuda(
    const Tensor& c0,
    const Tensor& c1,
    const Tensor& scalar,
    const Tensor& mod,
    const Tensor& barrett_mu,
    int64_t cur_limbs) {
  return fused_pair_arithmetic<ArithmeticOp::Mul, FusedRhsLayout::ScalarByLimb, true>(
      c0, c1, scalar, nullptr, mod, &barrett_mu, cur_limbs);
}

Tensor& cv_mul_scalar_pair_inplace_cuda(
    Tensor& c0,
    Tensor& c1,
    const Tensor& scalar,
    const Tensor& mod,
    const Tensor& barrett_mu,
    int64_t cur_limbs) {
  fused_pair_arithmetic_out<ArithmeticOp::Mul, FusedRhsLayout::ScalarByLimb, true>(
      c0, c1, c0, c1, scalar, nullptr, mod, &barrett_mu, cur_limbs);
  return c0;
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

} // namespace at::native
