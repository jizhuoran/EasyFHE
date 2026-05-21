#include <ATen/Dispatch_v2.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/interned_strings.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/empty.h>

#include <optional>

#include "ATen/native/fhe/cuda/CommonOperation.h"
#include "ATen/native/fhe/cuda/Utils.cuh"

#pragma clang diagnostic ignored "-Wmissing-prototypes"

namespace fhe {

enum class HMulPostOp : int {
  None = 0,
  AddCipher = 1,
  SubCipher = 2,
  AddScalar = 3,
  AddPlain = 4,
};

__global__ void hmul_raw_kernel(
    uint64_t* __restrict__ raw,
    const uint64_t* __restrict__ c0,
    const uint64_t* __restrict__ c1,
    const uint64_t* __restrict__ d0,
    const uint64_t* __restrict__ d1,
    const uint64_t* __restrict__ primes,
    const uint64_t* __restrict__ q_mu,
    const int64_t curr_limbs,
    const int64_t N) {
  const int64_t coeff = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  if (coeff >= N) {
    return;
  }
  const int64_t limb = blockIdx.y;
  const int64_t idx = limb * N + coeff;
  const int64_t LN = curr_limbs * N;

  const uint64_t prime = primes[limb];
  const uint64_t mu0 = q_mu[limb * 2];
  const uint64_t mu1 = q_mu[limb * 2 + 1];

  const uint64_t a0 = c0[idx];
  const uint64_t a1 = c1[idx];
  const uint64_t b0 = d0[idx];
  const uint64_t b1 = d1[idx];

  const uint64_t bx = mul_mod(a0, b0, prime, mu0, mu1);
  uint64_t ax = mul_mod(a0, b1, prime, mu0, mu1);
  const uint64_t ax_other = mul_mod(a1, b0, prime, mu0, mu1);
  ax = add_mod(ax, ax_other, prime);
  const uint64_t axax = mul_mod(a1, b1, prime, mu0, mu1);

  raw[idx] = bx;
  raw[LN + idx] = ax;
  raw[2 * LN + idx] = axax;
}

template <bool APPLY_DOUBLE>
__global__ void hmul_relin_scale_last_limb_kernel(
    uint64_t* __restrict__ out_last,
    const uint64_t* __restrict__ raw,
    const uint64_t* __restrict__ inner_product,
    const uint64_t* __restrict__ moddown_base,
    const uint64_t* __restrict__ prod_inv,
    const uint64_t* __restrict__ prod_inv_shoup,
    const uint64_t* __restrict__ primes,
    const int64_t curr_limbs,
    const int64_t sizeP,
    const int64_t N) {
  const int64_t coeff = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  if (coeff >= N) {
    return;
  }
  const int64_t cv = blockIdx.y;
  const int64_t limb = curr_limbs - 1;
  const int64_t idx = limb * N + coeff;
  const int64_t LN = curr_limbs * N;
  const int64_t L_INN = (curr_limbs + sizeP) * N;
  const uint64_t prime = primes[limb];

  uint64_t key = sub_mod(inner_product[cv * L_INN + idx], moddown_base[cv * LN + idx], prime);
  key = mul_and_reduce_shoup(key, prod_inv[limb], prod_inv_shoup[limb], prime);
  if (key >= prime) {
    key -= prime;
  }

  uint64_t value = add_mod(raw[cv * LN + idx], key, prime);
  if constexpr (APPLY_DOUBLE) {
    value = add_mod(value, value, prime);
  }
  out_last[cv * N + coeff] = value;
}

template <bool APPLY_DOUBLE, HMulPostOp POST_OP>
__global__ void hmul_const_mult_add_relin_scale_kernel(
    uint64_t* __restrict__ out,
    const uint64_t* __restrict__ raw,
    const uint64_t* __restrict__ inner_product,
    const uint64_t* __restrict__ moddown_base,
    const uint64_t* __restrict__ prod_inv,
    const uint64_t* __restrict__ prod_inv_shoup,
    const uint64_t* __restrict__ cnst,
    const uint64_t* __restrict__ cnst_shoup,
    const uint64_t* __restrict__ post_c0,
    const uint64_t* __restrict__ post_c1,
    const uint64_t* __restrict__ post_scalar,
    const uint64_t* __restrict__ primes,
    const int64_t curr_limbs,
    const int64_t sizeP,
    const int64_t N) {
  const int64_t coeff = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  if (coeff >= N) {
    return;
  }
  const int64_t limb = blockIdx.y;
  const int64_t cv = blockIdx.z;
  const int64_t end_length = curr_limbs - 1;
  const int64_t in_idx = limb * N + coeff;
  const int64_t in_LN = curr_limbs * N;
  const int64_t inner_LN = (curr_limbs + sizeP) * N;
  const int64_t out_idx = cv * end_length * N + in_idx;
  const uint64_t prime = primes[limb];

  uint64_t key = sub_mod(
      inner_product[cv * inner_LN + in_idx],
      moddown_base[cv * in_LN + in_idx],
      prime);
  key = mul_and_reduce_shoup(key, prod_inv[limb], prod_inv_shoup[limb], prime);
  if (key >= prime) {
    key -= prime;
  }

  uint64_t value = add_mod(raw[cv * in_LN + in_idx], key, prime);
  if constexpr (APPLY_DOUBLE) {
    value = add_mod(value, value, prime);
  }

  uint64_t scaled =
      mul_and_reduce_shoup(value, cnst[limb], cnst_shoup[limb], prime);
  if (scaled >= prime) {
    scaled -= prime;
  }
  uint64_t result = add_mod(out[out_idx], scaled, prime);
  if constexpr (POST_OP == HMulPostOp::AddCipher) {
    const uint64_t* post = (cv == 0) ? post_c0 : post_c1;
    result = add_mod(result, post[in_idx], prime);
  } else if constexpr (POST_OP == HMulPostOp::SubCipher) {
    const uint64_t* post = (cv == 0) ? post_c0 : post_c1;
    result = sub_mod(result, post[in_idx], prime);
  } else if constexpr (POST_OP == HMulPostOp::AddScalar) {
    if (cv == 0) {
      result = add_mod(result, post_scalar[limb], prime);
    }
  } else if constexpr (POST_OP == HMulPostOp::AddPlain) {
    if (cv == 0) {
      result = add_mod(result, post_c0[in_idx], prime);
    }
  }
  out[out_idx] = result;
}

__global__ void hmul_moddown_base_convert_kernel(
    uint64_t* __restrict__ to,
    const uint64_t* __restrict__ from,
    const int64_t N,
    const int64_t L_OUTN,
    const int64_t L_INN,
    const int64_t L,
    const int64_t sizeP,
    const uint64_t* __restrict__ hat_inverse,
    const uint64_t* __restrict__ hat_inverse_shoup,
    const uint64_t* __restrict__ hat_mod_end,
    const uint64_t* __restrict__ primes,
    const uint64_t* __restrict__ barret_ratios,
    const uint64_t* __restrict__ barret_ks) {
  const int64_t degree_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (degree_idx >= N) {
    return;
  }

  __shared__ uint64_t hat_mod_end_shared[997];
  if (threadIdx.x < sizeP) {
    hat_mod_end_shared[threadIdx.x] =
        hat_mod_end[threadIdx.x + blockIdx.y * sizeP];
  }
  __syncthreads();

  const int64_t out_idx = blockIdx.y;
  const int64_t cv_id = blockIdx.z;
  const uint64_t* from_cv = from + cv_id * L_INN;
  uint64_t* to_cv = to + cv_id * L_OUTN;

  uint128_t accum{0};
  for (int64_t i = 0; i < sizeP; ++i) {
    const uint64_t p_prime = primes[L + i];
    uint64_t op1 = mul_and_reduce_shoup(
        from_cv[i * N + degree_idx],
        hat_inverse[i],
        hat_inverse_shoup[i],
        p_prime);
    if (op1 >= p_prime) {
      op1 -= p_prime;
    }
    const uint64_t op2 = hat_mod_end_shared[i];
    uint128_t product = mult_64_64_128(op1, op2);
    inplace_add_128_128(product, accum);
  }

  const uint64_t prime = primes[out_idx];
  const uint64_t barret_ratio = barret_ratios[out_idx];
  const uint64_t barret_k = barret_ks[out_idx];
  to_cv[out_idx * N + degree_idx] =
      barret_reduction_128_64(accum, prime, barret_ratio, barret_k);
}

__global__ void hmul_switch_modulus_const_mult_kernel(
    uint64_t* __restrict__ to,
    const uint64_t* __restrict__ from_last,
    const uint64_t* __restrict__ cnst,
    const uint64_t* __restrict__ cnst_shoup,
    const uint64_t* __restrict__ primes,
    const uint64_t* __restrict__ diffs,
    const uint64_t old_modulus_by_two,
    const int64_t N,
    const int64_t curr_limbs) {
  const int64_t coeff = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  if (coeff >= N) {
    return;
  }
  const int64_t limb = blockIdx.y;
  const int64_t cv = blockIdx.z;
  const int64_t end_length = curr_limbs - 1;
  const int64_t in_LN = curr_limbs * N;
  const int64_t out_LN = end_length * N;
  const uint64_t prime = primes[limb];

  const uint64_t in_val = from_last[cv * in_LN + coeff];
  uint64_t switched =
      in_val + (in_val > old_modulus_by_two ? diffs[limb] : uint64_t{0});
  if (switched >= prime) {
    switched -= prime;
  }

  uint64_t scaled =
      mul_and_reduce_shoup(switched, cnst[limb], cnst_shoup[limb], prime);
  if (scaled >= prime) {
    scaled -= prime;
  }
  to[cv * out_LN + limb * N + coeff] = scaled;
}

__global__ void hmul_innerproduct_without_original_copy_kernel(
    uint64_t* __restrict__ out,
    const uint64_t* __restrict__ in_modup,
    const uint64_t* __restrict__ raw_axax,
    const uint64_t* __restrict__ eval_bx,
    const uint64_t* __restrict__ eval_ax,
    const uint64_t* __restrict__ primes,
    const uint64_t* __restrict__ barret_ratios,
    const uint64_t* __restrict__ barret_ks,
    const int64_t N,
    const int64_t length,
    const int64_t mult_length,
    const int64_t beta,
    const int64_t curr_limbs,
    const int64_t alpha,
    const int64_t prime_gap,
    const int64_t special_mod_start) {
  const int64_t coeff = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  if (coeff >= N) {
    return;
  }
  const int64_t limb = blockIdx.y;
  const int64_t i = limb * N + coeff;
  const int64_t swk_gap = special_mod_start - curr_limbs;
  const int64_t prime_idx = (limb < curr_limbs) ? 0 : prime_gap;
  const int64_t swk_idx = (limb < curr_limbs) ? 0 : swk_gap;
  const int64_t reduce_prime_idx = limb + prime_idx;
  const uint64_t prime = primes[reduce_prime_idx];
  const uint64_t barret_ratio = barret_ratios[reduce_prime_idx];
  const uint64_t barret_k = barret_ks[reduce_prime_idx];

  uint128_t accum_ax{0};
  uint128_t accum_bx{0};
  for (int64_t beta_idx = 0; beta_idx < beta; ++beta_idx) {
    const int64_t begin_idx = beta_idx * alpha;
    const int64_t group_size = min(alpha, curr_limbs - begin_idx);
    const bool is_original_limb =
        limb >= begin_idx && limb < begin_idx + group_size;
    const int64_t in_stride = N * length * beta_idx;
    const int64_t swk_stride = N * (mult_length * beta_idx + swk_idx);
    const uint64_t op1 =
        is_original_limb ? raw_axax[i] : in_modup[i + in_stride];
    const auto mul_ax = mult_64_64_128(op1, eval_ax[i + swk_stride]);
    const auto mul_bx = mult_64_64_128(op1, eval_bx[i + swk_stride]);
    inplace_add_128_128(mul_ax, accum_ax);
    inplace_add_128_128(mul_bx, accum_bx);
  }

  out[i] = barret_reduction_128_64(accum_bx, prime, barret_ratio, barret_k);
  out[length * N + i] =
      barret_reduction_128_64(accum_ax, prime, barret_ratio, barret_k);
}

} // namespace fhe

namespace at::native {

template <bool APPLY_DOUBLE>
static void launch_hmul_relin_scale_last_limb(
    uint64_t* out_last,
    const uint64_t* raw,
    const uint64_t* inner_ptr,
    const uint64_t* moddown_base_ptr,
    const uint64_t* prod_inv,
    const uint64_t* prod_inv_shoup,
    const uint64_t* primes,
    int64_t curr_limbs,
    int64_t sizeP,
    int64_t N,
    cudaStream_t stream) {
  fhe::hmul_relin_scale_last_limb_kernel<APPLY_DOUBLE>
      <<<dim3(num_blocks(N), 2), BLOCK_SIZE, 0, stream>>>(
          out_last,
          raw,
          inner_ptr,
          moddown_base_ptr,
          prod_inv,
          prod_inv_shoup,
          primes,
          curr_limbs,
          sizeP,
          N);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool APPLY_DOUBLE, fhe::HMulPostOp POST_OP>
static void launch_hmul_const_mult_add_relin_scale(
    uint64_t* out_ptr,
    const uint64_t* raw,
    const uint64_t* inner_ptr,
    const uint64_t* moddown_base_ptr,
    const uint64_t* prod_inv,
    const uint64_t* prod_inv_shoup,
    const uint64_t* cnst,
    const uint64_t* cnst_shoup,
    const uint64_t* post_c0,
    const uint64_t* post_c1,
    const uint64_t* post_scalar,
    const uint64_t* primes,
    int64_t curr_limbs,
    int64_t sizeP,
    int64_t N,
    cudaStream_t stream) {
  fhe::hmul_const_mult_add_relin_scale_kernel<APPLY_DOUBLE, POST_OP>
      <<<dim3(num_blocks(N), curr_limbs - 1, 2), BLOCK_SIZE, 0, stream>>>(
          out_ptr,
          raw,
          inner_ptr,
          moddown_base_ptr,
          prod_inv,
          prod_inv_shoup,
          cnst,
          cnst_shoup,
          post_c0,
          post_c1,
          post_scalar,
          primes,
          curr_limbs,
          sizeP,
          N);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool APPLY_DOUBLE>
static void dispatch_hmul_const_mult_add_relin_scale(
    int64_t post_op,
    uint64_t* out_ptr,
    const uint64_t* raw,
    const uint64_t* inner_ptr,
    const uint64_t* moddown_base_ptr,
    const uint64_t* prod_inv,
    const uint64_t* prod_inv_shoup,
    const uint64_t* cnst,
    const uint64_t* cnst_shoup,
    const uint64_t* post_c0,
    const uint64_t* post_c1,
    const uint64_t* post_scalar,
    const uint64_t* primes,
    int64_t curr_limbs,
    int64_t sizeP,
    int64_t N,
    cudaStream_t stream) {
  switch (post_op) {
    case static_cast<int64_t>(fhe::HMulPostOp::None):
      launch_hmul_const_mult_add_relin_scale<APPLY_DOUBLE, fhe::HMulPostOp::None>(
          out_ptr,
          raw,
          inner_ptr,
          moddown_base_ptr,
          prod_inv,
          prod_inv_shoup,
          cnst,
          cnst_shoup,
          post_c0,
          post_c1,
          post_scalar,
          primes,
          curr_limbs,
          sizeP,
          N,
          stream);
      return;
    case static_cast<int64_t>(fhe::HMulPostOp::AddCipher):
      launch_hmul_const_mult_add_relin_scale<APPLY_DOUBLE, fhe::HMulPostOp::AddCipher>(
          out_ptr,
          raw,
          inner_ptr,
          moddown_base_ptr,
          prod_inv,
          prod_inv_shoup,
          cnst,
          cnst_shoup,
          post_c0,
          post_c1,
          post_scalar,
          primes,
          curr_limbs,
          sizeP,
          N,
          stream);
      return;
    case static_cast<int64_t>(fhe::HMulPostOp::SubCipher):
      launch_hmul_const_mult_add_relin_scale<APPLY_DOUBLE, fhe::HMulPostOp::SubCipher>(
          out_ptr,
          raw,
          inner_ptr,
          moddown_base_ptr,
          prod_inv,
          prod_inv_shoup,
          cnst,
          cnst_shoup,
          post_c0,
          post_c1,
          post_scalar,
          primes,
          curr_limbs,
          sizeP,
          N,
          stream);
      return;
    case static_cast<int64_t>(fhe::HMulPostOp::AddScalar):
      launch_hmul_const_mult_add_relin_scale<APPLY_DOUBLE, fhe::HMulPostOp::AddScalar>(
          out_ptr,
          raw,
          inner_ptr,
          moddown_base_ptr,
          prod_inv,
          prod_inv_shoup,
          cnst,
          cnst_shoup,
          post_c0,
          post_c1,
          post_scalar,
          primes,
          curr_limbs,
          sizeP,
          N,
          stream);
      return;
    case static_cast<int64_t>(fhe::HMulPostOp::AddPlain):
      launch_hmul_const_mult_add_relin_scale<APPLY_DOUBLE, fhe::HMulPostOp::AddPlain>(
          out_ptr,
          raw,
          inner_ptr,
          moddown_base_ptr,
          prod_inv,
          prod_inv_shoup,
          cnst,
          cnst_shoup,
          post_c0,
          post_c1,
          post_scalar,
          primes,
          curr_limbs,
          sizeP,
          N,
          stream);
      return;
    default:
      TORCH_CHECK(false, "unsupported hmul post_op: ", post_op);
  }
}

static Tensor hmul_innerproduct_without_original_copy(
    const Tensor& modup,
    const Tensor& raw,
    const Tensor& swk_bx,
    const Tensor& swk_ax,
    int64_t curr_limbs,
    int64_t alpha,
    int64_t special_mod_start,
    int64_t L,
    int64_t sizeP,
    int64_t N,
    const Tensor& primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k) {
  const int64_t beta = (curr_limbs + alpha - 1) / alpha;
  const int64_t length = curr_limbs + sizeP;
  const int64_t mult_length = special_mod_start + sizeP;
  const int64_t prime_gap = L - curr_limbs;

  auto out = at::empty({2, 1, length, N}, raw.options());
  const int64_t raw_LN = curr_limbs * N;
  const auto stream = at::cuda::getCurrentCUDAStream();
  fhe::hmul_innerproduct_without_original_copy_kernel<<<
      dim3(num_blocks(N), length),
      BLOCK_SIZE,
      0,
      stream>>>(
      out.data_ptr<uint64_t>(),
      modup.data_ptr<uint64_t>(),
      raw.data_ptr<uint64_t>() + 2 * raw_LN,
      swk_bx.data_ptr<uint64_t>(),
      swk_ax.data_ptr<uint64_t>(),
      primes.data_ptr<uint64_t>(),
      barret_ratio.data_ptr<uint64_t>(),
      barret_k.data_ptr<uint64_t>(),
      N,
      length,
      mult_length,
      beta,
      curr_limbs,
      alpha,
      prime_gap,
      special_mod_start);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

static Tensor hmul_moddown_drop_last_scale(
    const Tensor& raw,
    const Tensor& inner_product,
    const std::optional<Tensor>& post_c0,
    const std::optional<Tensor>& post_c1,
    const std::optional<Tensor>& post_scalar,
    int64_t curr_limbs,
    int64_t L,
    int64_t sizeP,
    int64_t N,
    int64_t old_prime,
    bool apply_double,
    int64_t post_op,
    const Tensor& hat_inverse_vec_moddown,
    const Tensor& hat_inverse_vec_shoup_moddown,
    const Tensor& prod_q_i_mod_q_j_moddown,
    const Tensor& prod_inv_moddown,
    const Tensor& prod_inv_shoup_moddown,
    const Tensor& primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    const Tensor& switch_modulus_map,
    const Tensor& power_of_roots_shoup,
    const Tensor& power_of_roots,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& qlql_inv_mod_ql_div_ql_mod_q,
    const Tensor& qlql_inv_mod_ql_div_ql_mod_q_shoup,
    const Tensor& q_inv_mod_q,
    const Tensor& q_inv_mod_q_shoup) {
  constexpr int64_t num_cv = 2;
  constexpr int64_t num_cipher = 1;
  const int64_t end_length = curr_limbs - 1;
  if (post_op == static_cast<int64_t>(fhe::HMulPostOp::AddCipher) ||
      post_op == static_cast<int64_t>(fhe::HMulPostOp::SubCipher)) {
    TORCH_CHECK(post_c0.has_value(), "hmul post cipher op requires post_c0/post_c1");
    TORCH_CHECK(!post_scalar.has_value(), "hmul post cipher op cannot also use post_scalar");
    TORCH_CHECK(post_c0->is_contiguous(), "hmul post_c0 must be contiguous");
    TORCH_CHECK(post_c1->is_contiguous(), "hmul post_c1 must be contiguous");
    TORCH_CHECK(post_c0->dim() == 2 && post_c1->dim() == 2, "hmul post cipher must be [limbs, N]");
    TORCH_CHECK(post_c0->sizes()[0] == end_length && post_c0->sizes()[1] == N, "hmul post_c0 shape mismatch");
    TORCH_CHECK(post_c1->sizes() == post_c0->sizes(), "hmul post_c1 shape mismatch");
  } else if (post_op == static_cast<int64_t>(fhe::HMulPostOp::AddScalar)) {
    TORCH_CHECK(post_scalar.has_value(), "hmul add scalar op requires post_scalar");
    TORCH_CHECK(!post_c0.has_value() && !post_c1.has_value(), "hmul scalar op cannot also use post cipher");
    TORCH_CHECK(post_scalar->is_contiguous(), "hmul post_scalar must be contiguous");
    TORCH_CHECK(post_scalar->dim() == 1 && post_scalar->sizes()[0] == end_length, "hmul post_scalar shape mismatch");
  } else if (post_op == static_cast<int64_t>(fhe::HMulPostOp::AddPlain)) {
    TORCH_CHECK(post_c0.has_value(), "hmul add plaintext op requires post_c0");
    TORCH_CHECK(!post_c1.has_value(), "hmul add plaintext op cannot use post_c1");
    TORCH_CHECK(!post_scalar.has_value(), "hmul add plaintext op cannot also use post_scalar");
    TORCH_CHECK(post_c0->is_contiguous(), "hmul post plaintext must be contiguous");
    TORCH_CHECK(post_c0->dim() == 2, "hmul post plaintext must be [limbs, N]");
    TORCH_CHECK(post_c0->sizes()[0] == end_length && post_c0->sizes()[1] == N, "hmul post plaintext shape mismatch");
  } else {
    TORCH_CHECK(!post_c0.has_value() && !post_c1.has_value(), "hmul no-post op got post cipher");
    TORCH_CHECK(!post_scalar.has_value(), "hmul no-post op got post scalar");
  }

  auto out = at::empty({num_cv, num_cipher, end_length, N}, raw.options());
  auto moddown_base = at::empty({num_cv, num_cipher, curr_limbs, N}, raw.options());
  auto moddown_workspace =
      at::empty({num_cv, num_cipher, curr_limbs + sizeP, N}, raw.options());
  auto workspace = at::empty({num_cv, num_cipher, curr_limbs, N}, raw.options());
  auto last_limb_ntt = at::empty({num_cv, num_cipher, 1, N}, raw.options());

  const dim3 block(BLOCK_SIZE);
  const auto stream = at::cuda::getCurrentCUDAStream();

  auto* inner_ptr = inner_product.data_ptr<uint64_t>();
  auto* moddown_workspace_ptr = moddown_workspace.data_ptr<uint64_t>();
  auto* moddown_base_ptr = moddown_base.data_ptr<uint64_t>();

  iNTT_impl(
      moddown_workspace_ptr + curr_limbs * N,
      inner_ptr + curr_limbs * N,
      sizeP,
      N,
      curr_limbs + sizeP,
      curr_limbs + sizeP,
      num_cv,
      num_cipher,
      primes.data_ptr<uint64_t>() + L,
      inverse_power_of_roots_div_two.data_ptr<uint64_t>() + L * N,
      inverse_scaled_power_of_roots_div_two.data_ptr<uint64_t>() + L * N);

  fhe::hmul_moddown_base_convert_kernel<<<
      dim3(num_blocks(N), curr_limbs, num_cv),
      block,
      0,
      stream>>>(
      moddown_base_ptr,
      moddown_workspace_ptr + curr_limbs * N,
      N,
      curr_limbs * N,
      (curr_limbs + sizeP) * N,
      L,
      sizeP,
      hat_inverse_vec_moddown.data_ptr<uint64_t>(),
      hat_inverse_vec_shoup_moddown.data_ptr<uint64_t>(),
      prod_q_i_mod_q_j_moddown.data_ptr<uint64_t>(),
      primes.data_ptr<uint64_t>(),
      barret_ratio.data_ptr<uint64_t>(),
      barret_k.data_ptr<uint64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  NTT_impl(
      moddown_base_ptr,
      curr_limbs,
      N,
      curr_limbs,
      num_cv,
      num_cipher,
      primes.data_ptr<uint64_t>(),
      power_of_roots_shoup.data_ptr<uint64_t>(),
      power_of_roots.data_ptr<uint64_t>());

  if (apply_double) {
    launch_hmul_relin_scale_last_limb<true>(
        last_limb_ntt.data_ptr<uint64_t>(),
        raw.data_ptr<uint64_t>(),
        inner_ptr,
        moddown_base_ptr,
        prod_inv_moddown.data_ptr<uint64_t>(),
        prod_inv_shoup_moddown.data_ptr<uint64_t>(),
        primes.data_ptr<uint64_t>(),
        curr_limbs,
        sizeP,
        N,
        stream);
  } else {
    launch_hmul_relin_scale_last_limb<false>(
        last_limb_ntt.data_ptr<uint64_t>(),
        raw.data_ptr<uint64_t>(),
        inner_ptr,
        moddown_base_ptr,
        prod_inv_moddown.data_ptr<uint64_t>(),
        prod_inv_shoup_moddown.data_ptr<uint64_t>(),
        primes.data_ptr<uint64_t>(),
        curr_limbs,
        sizeP,
        N,
        stream);
  }

  auto* workspace_ptr = workspace.data_ptr<uint64_t>();
  auto* out_ptr = out.data_ptr<uint64_t>();

  iNTT_impl(
      workspace_ptr + N * end_length,
      last_limb_ntt.data_ptr<uint64_t>(),
      1,
      N,
      curr_limbs,
      1,
      num_cv,
      num_cipher,
      primes.data_ptr<uint64_t>() + end_length,
      inverse_power_of_roots_div_two.data_ptr<uint64_t>() + end_length * N,
      inverse_scaled_power_of_roots_div_two.data_ptr<uint64_t>() +
          end_length * N);

  int64_t start_op2_idx = (L - curr_limbs) * (L - 1);
  fhe::hmul_switch_modulus_const_mult_kernel<<<
      dim3(num_blocks(N), end_length, num_cv),
      block,
      0,
      stream>>>(
      out_ptr,
      workspace_ptr + N * end_length,
      qlql_inv_mod_ql_div_ql_mod_q.data_ptr<uint64_t>() + start_op2_idx,
      qlql_inv_mod_ql_div_ql_mod_q_shoup.data_ptr<uint64_t>() + start_op2_idx,
      primes.data_ptr<uint64_t>(),
      switch_modulus_map.data_ptr<uint64_t>() + end_length * primes.numel(),
      static_cast<uint64_t>(old_prime) >> 1,
      N,
      curr_limbs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  NTT_impl(
      out_ptr,
      end_length,
      N,
      end_length,
      num_cv,
      num_cipher,
      primes.data_ptr<uint64_t>(),
      power_of_roots_shoup.data_ptr<uint64_t>(),
      power_of_roots.data_ptr<uint64_t>());

  start_op2_idx = end_length * L;
  const uint64_t* post_c0_ptr =
      post_c0.has_value() ? post_c0->data_ptr<uint64_t>() : nullptr;
  const uint64_t* post_c1_ptr =
      post_c1.has_value() ? post_c1->data_ptr<uint64_t>() : nullptr;
  const uint64_t* post_scalar_ptr =
      post_scalar.has_value() ? post_scalar->data_ptr<uint64_t>() : nullptr;
  if (apply_double) {
    dispatch_hmul_const_mult_add_relin_scale<true>(
        post_op,
        out_ptr,
        raw.data_ptr<uint64_t>(),
        inner_ptr,
        moddown_base_ptr,
        prod_inv_moddown.data_ptr<uint64_t>(),
        prod_inv_shoup_moddown.data_ptr<uint64_t>(),
        q_inv_mod_q.data_ptr<uint64_t>() + start_op2_idx,
        q_inv_mod_q_shoup.data_ptr<uint64_t>() + start_op2_idx,
        post_c0_ptr,
        post_c1_ptr,
        post_scalar_ptr,
        primes.data_ptr<uint64_t>(),
        curr_limbs,
        sizeP,
        N,
        stream);
  } else {
    dispatch_hmul_const_mult_add_relin_scale<false>(
        post_op,
        out_ptr,
        raw.data_ptr<uint64_t>(),
        inner_ptr,
        moddown_base_ptr,
        prod_inv_moddown.data_ptr<uint64_t>(),
        prod_inv_shoup_moddown.data_ptr<uint64_t>(),
        q_inv_mod_q.data_ptr<uint64_t>() + start_op2_idx,
        q_inv_mod_q_shoup.data_ptr<uint64_t>() + start_op2_idx,
        post_c0_ptr,
        post_c1_ptr,
        post_scalar_ptr,
        primes.data_ptr<uint64_t>(),
        curr_limbs,
        sizeP,
        N,
        stream);
  }

  return out;
}

static Tensor hmul_double_rescale_impl(
    const Tensor& c0,
    const Tensor& c1,
    const Tensor& d0,
    const Tensor& d1,
    const Tensor& swk_bx,
    const Tensor& swk_ax,
    const std::optional<Tensor>& post_c0,
    const std::optional<Tensor>& post_c1,
    const std::optional<Tensor>& post_scalar,
    int64_t curr_limbs,
    int64_t special_mod_start,
    int64_t L,
    int64_t beta,
    int64_t N,
    int64_t alpha,
    int64_t old_prime,
    const Tensor& primes,
    const Tensor& q_mu,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    const Tensor& hat_inverse_vec_modup,
    const Tensor& hat_inverse_vec_shoup_modup,
    const Tensor& prod_q_i_mod_q_j_modup,
    const Tensor& hat_inverse_vec_moddown,
    const Tensor& hat_inverse_vec_shoup_moddown,
    const Tensor& prod_q_i_mod_q_j_moddown,
    const Tensor& prod_inv_moddown,
    const Tensor& prod_inv_shoup_moddown,
    const Tensor& power_of_roots_shoup,
    const Tensor& power_of_roots,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& switch_modulus_map,
    const Tensor& qlql_inv_mod_ql_div_ql_mod_q,
    const Tensor& qlql_inv_mod_ql_div_ql_mod_q_shoup,
    const Tensor& q_inv_mod_q,
    const Tensor& q_inv_mod_q_shoup,
    const Tensor& inner_workspace,
    bool apply_double,
    int64_t post_op) {
  TORCH_CHECK(c0.is_contiguous(), "hmul c0 must be contiguous");
  TORCH_CHECK(c1.is_contiguous(), "hmul c1 must be contiguous");
  TORCH_CHECK(d0.is_contiguous(), "hmul d0 must be contiguous");
  TORCH_CHECK(d1.is_contiguous(), "hmul d1 must be contiguous");
  TORCH_CHECK(c0.dim() == 2 && c1.dim() == 2, "hmul inputs must be [limbs, N]");
  TORCH_CHECK(c0.size(0) == curr_limbs && c0.size(1) == N, "hmul c0 shape mismatch");
  TORCH_CHECK(c1.sizes() == c0.sizes(), "hmul c1 shape mismatch");
  TORCH_CHECK(d0.sizes() == c0.sizes(), "hmul d0 shape mismatch");
  TORCH_CHECK(d1.sizes() == c0.sizes(), "hmul d1 shape mismatch");

  auto raw = at::empty({3, 1, curr_limbs, N}, c0.options());
  const dim3 block(BLOCK_SIZE);
  const dim3 grid(num_blocks(N), curr_limbs);
  const auto stream = at::cuda::getCurrentCUDAStream();

  fhe::hmul_raw_kernel<<<grid, block, 0, stream>>>(
      raw.data_ptr<uint64_t>(),
      c0.data_ptr<uint64_t>(),
      c1.data_ptr<uint64_t>(),
      d0.data_ptr<uint64_t>(),
      d1.data_ptr<uint64_t>(),
      primes.data_ptr<uint64_t>(),
      q_mu.data_ptr<uint64_t>(),
      curr_limbs,
      N);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int64_t sizeP = primes.numel() - L;
  const auto axax = raw.slice(0, 2, 3);
  const auto modup = modup_without_copy_cuda(
      axax,
      curr_limbs,
      L,
      beta,
      N,
      alpha,
      hat_inverse_vec_modup,
      hat_inverse_vec_shoup_modup,
      prod_q_i_mod_q_j_modup,
      primes,
      barret_ratio,
      barret_k,
      power_of_roots_shoup,
      power_of_roots,
      inverse_power_of_roots_div_two,
      inverse_scaled_power_of_roots_div_two);

  const auto inner_product = hmul_innerproduct_without_original_copy(
      modup,
      raw,
      swk_bx,
      swk_ax,
      curr_limbs,
      alpha,
      special_mod_start,
      L,
      sizeP,
      N,
      primes,
      barret_ratio,
      barret_k);

  return hmul_moddown_drop_last_scale(
      raw,
      inner_product,
      post_c0,
      post_c1,
      post_scalar,
      curr_limbs,
      L,
      sizeP,
      N,
      old_prime,
      apply_double,
      post_op,
      hat_inverse_vec_moddown,
      hat_inverse_vec_shoup_moddown,
      prod_q_i_mod_q_j_moddown,
      prod_inv_moddown,
      prod_inv_shoup_moddown,
      primes,
      barret_ratio,
      barret_k,
      switch_modulus_map,
      power_of_roots_shoup,
      power_of_roots,
      inverse_power_of_roots_div_two,
      inverse_scaled_power_of_roots_div_two,
      qlql_inv_mod_ql_div_ql_mod_q,
      qlql_inv_mod_ql_div_ql_mod_q_shoup,
      q_inv_mod_q,
      q_inv_mod_q_shoup);
}

Tensor hmul_double_rescale_cuda(
    const Tensor& c0,
    const Tensor& c1,
    const Tensor& d0,
    const Tensor& d1,
    const Tensor& swk_bx,
    const Tensor& swk_ax,
    int64_t curr_limbs,
    int64_t special_mod_start,
    int64_t L,
    int64_t beta,
    int64_t N,
    int64_t alpha,
    int64_t old_prime,
    const Tensor& primes,
    const Tensor& q_mu,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    const Tensor& hat_inverse_vec_modup,
    const Tensor& hat_inverse_vec_shoup_modup,
    const Tensor& prod_q_i_mod_q_j_modup,
    const Tensor& hat_inverse_vec_moddown,
    const Tensor& hat_inverse_vec_shoup_moddown,
    const Tensor& prod_q_i_mod_q_j_moddown,
    const Tensor& prod_inv_moddown,
    const Tensor& prod_inv_shoup_moddown,
    const Tensor& power_of_roots_shoup,
    const Tensor& power_of_roots,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& switch_modulus_map,
    const Tensor& qlql_inv_mod_ql_div_ql_mod_q,
    const Tensor& qlql_inv_mod_ql_div_ql_mod_q_shoup,
    const Tensor& q_inv_mod_q,
    const Tensor& q_inv_mod_q_shoup,
    const Tensor& inner_workspace,
    bool apply_double,
    int64_t post_op,
    const std::optional<Tensor>& post_c0,
    const std::optional<Tensor>& post_c1,
    const std::optional<Tensor>& post_scalar) {
  return hmul_double_rescale_impl(
      c0,
      c1,
      d0,
      d1,
      swk_bx,
      swk_ax,
      post_c0,
      post_c1,
      post_scalar,
      curr_limbs,
      special_mod_start,
      L,
      beta,
      N,
      alpha,
      old_prime,
      primes,
      q_mu,
      barret_ratio,
      barret_k,
      hat_inverse_vec_modup,
      hat_inverse_vec_shoup_modup,
      prod_q_i_mod_q_j_modup,
      hat_inverse_vec_moddown,
      hat_inverse_vec_shoup_moddown,
      prod_q_i_mod_q_j_moddown,
      prod_inv_moddown,
      prod_inv_shoup_moddown,
      power_of_roots_shoup,
      power_of_roots,
      inverse_power_of_roots_div_two,
      inverse_scaled_power_of_roots_div_two,
      switch_modulus_map,
      qlql_inv_mod_ql_div_ql_mod_q,
      qlql_inv_mod_ql_div_ql_mod_q_shoup,
      q_inv_mod_q,
      q_inv_mod_q_shoup,
      inner_workspace,
      apply_double,
      post_op);
}

} // namespace at::native
