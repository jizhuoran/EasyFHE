#include <ATen/core/Tensor.h>
#include <ATen/core/TensorBody.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/thread_constants.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/reshape.h>

#include <vector>

#include "ATen/native/fhe/cuda/CommonOperation.h"
#include "ATen/native/fhe/cuda/Utils.cuh"

#pragma clang diagnostic ignored "-Wmissing-prototypes"

namespace fhe {

__device__ __forceinline__ void hrot_butt_ntt_local(
    uint64_t& a,
    uint64_t& b,
    const uint64_t w,
    const uint64_t w_shoup,
    const uint64_t p,
    const uint64_t two_p) {
  uint64_t u = mul_and_reduce_shoup(b, w, w_shoup, p);
  if (a >= two_p) {
    a -= two_p;
  }
  b = a + (two_p - u);
  a += u;
}

template <uint8_t radix>
__device__ __forceinline__ void hrot_local_ntt_radix(
    uint64_t& local0,
    uint64_t& local1,
    uint64_t& local2,
    uint64_t& local3,
    uint64_t& local4,
    uint64_t& local5,
    uint64_t& local6,
    uint64_t& local7,
    const uint32_t tw_off,
    const uint64_t* __restrict__ W,
    const uint64_t* __restrict__ W_shoup,
    const uint64_t prime,
    const uint64_t two_p) {
  static_assert(radix == 2 || radix == 4 || radix == 8);
  if constexpr (radix >= 8) {
    const uint64_t w = W[tw_off];
    const uint64_t ws = W_shoup[tw_off];
    hrot_butt_ntt_local(local0, local4, w, ws, prime, two_p);
    hrot_butt_ntt_local(local1, local5, w, ws, prime, two_p);
    hrot_butt_ntt_local(local2, local6, w, ws, prime, two_p);
    hrot_butt_ntt_local(local3, local7, w, ws, prime, two_p);
  }

  if constexpr (radix >= 4) {
    const uint32_t off = 2 * tw_off;
    const uint64_t w0 = W[off];
    const uint64_t ws0 = W_shoup[off];
    const uint64_t w1 = W[off + 1];
    const uint64_t ws1 = W_shoup[off + 1];
    hrot_butt_ntt_local(local0, local2, w0, ws0, prime, two_p);
    hrot_butt_ntt_local(local1, local3, w0, ws0, prime, two_p);
    hrot_butt_ntt_local(local4, local6, w1, ws1, prime, two_p);
    hrot_butt_ntt_local(local5, local7, w1, ws1, prime, two_p);
  }

  if constexpr (radix >= 2) {
    const uint32_t off = 4 * tw_off;
    const uint64_t w0 = W[off];
    const uint64_t ws0 = W_shoup[off];
    const uint64_t w1 = W[off + 1];
    const uint64_t ws1 = W_shoup[off + 1];
    const uint64_t w2 = W[off + 2];
    const uint64_t ws2 = W_shoup[off + 2];
    const uint64_t w3 = W[off + 3];
    const uint64_t ws3 = W_shoup[off + 3];
    hrot_butt_ntt_local(local0, local1, w0, ws0, prime, two_p);
    hrot_butt_ntt_local(local2, local3, w1, ws1, prime, two_p);
    hrot_butt_ntt_local(local4, local5, w2, ws2, prime, two_p);
    hrot_butt_ntt_local(local6, local7, w3, ws3, prime, two_p);
  }
}

template <size_t LOG_N, size_t NUM_GROUPS>
__global__ void hrot_moddown_base_convert_ntt_phase1_kernel(
    uint64_t* __restrict__ workspace,
    const int64_t L_IN,
    const int64_t curr_limbs,
    const int64_t sizeP,
    const uint64_t* __restrict__ hat_mod_end,
    const uint64_t* __restrict__ primes,
    const uint64_t* __restrict__ barret_ratios,
    const uint64_t* __restrict__ barret_ks,
    const uint64_t* __restrict__ power_of_roots,
    const uint64_t* __restrict__ power_of_roots_shoup) {
  static_assert(NUM_GROUPS == 8);
  constexpr int GROUP_SIZE = 8;
  constexpr int C_N = 1 << LOG_N;
  constexpr int COEFF_STRIDE = C_N / (8 * GROUP_SIZE);

  const int groupID = threadIdx.x / GROUP_SIZE;
  const int laneID = threadIdx.x % GROUP_SIZE;
  const int64_t limb = blockIdx.y;
  const int64_t cv_id = blockIdx.z;
  const int N_init = NUM_GROUPS * blockIdx.x + laneID;

  __shared__ uint64_t hat_mod_end_shared[997];
  if (threadIdx.x < sizeP) {
    hat_mod_end_shared[threadIdx.x] =
        hat_mod_end[threadIdx.x + limb * sizeP];
  }
  __syncthreads();

  uint64_t* cv_workspace = workspace + cv_id * L_IN * C_N;
  const uint64_t* p_limbs = cv_workspace + curr_limbs * C_N;
  auto inout_matrix =
      reinterpret_cast<uint64_t(*)[8][GROUP_SIZE][COEFF_STRIDE]>(cv_workspace);

  power_of_roots += limb * C_N;
  power_of_roots_shoup += limb * C_N;
  const uint64_t prime = primes[limb];
  const uint64_t two_p = prime << 1;
  const uint64_t barret_ratio = barret_ratios[limb];
  const uint64_t barret_k = barret_ks[limb];

  uint64_t local[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int coeff = j * (GROUP_SIZE * COEFF_STRIDE) +
        groupID * COEFF_STRIDE + N_init;
    uint128_t accum{0};
    for (int i = 0; i < sizeP; i++) {
      const uint64_t op1 = p_limbs[i * C_N + coeff];
      const uint64_t op2 = hat_mod_end_shared[i];
      uint128_t out = mult_64_64_128(op1, op2);
      inplace_add_128_128(out, accum);
    }
    local[j] = barret_reduction_128_64(accum, prime, barret_ratio, barret_k);
  }

  hrot_local_ntt_radix<8>(
      local[0],
      local[1],
      local[2],
      local[3],
      local[4],
      local[5],
      local[6],
      local[7],
      1,
      power_of_roots,
      power_of_roots_shoup,
      prime,
      two_p);

  __shared__ uint64_t transpose_matrix[NUM_GROUPS][GROUP_SIZE + 1][8 + 1];

#pragma unroll
  for (int j = 0; j < 8; ++j) {
    transpose_matrix[laneID][j][groupID] = local[j];
  }
  __syncthreads();

#pragma unroll
  for (int l = 0; l < 8; ++l) {
    local[l] = transpose_matrix[laneID][groupID][l];
  }

  hrot_local_ntt_radix<8>(
      local[0],
      local[1],
      local[2],
      local[3],
      local[4],
      local[5],
      local[6],
      local[7],
      8 + groupID,
      power_of_roots,
      power_of_roots_shoup,
      prime,
      two_p);

#pragma unroll
  for (int j = 0; j < 8; ++j) {
    inout_matrix[limb][groupID][j][N_init] = local[j];
  }
}

__global__ void hrot_sum_reduce_from_modup_kernel(
    uint64_t* __restrict__ out_ax,
    uint64_t* __restrict__ out_bx,
    const uint64_t* __restrict__ in_modup,
    const uint64_t* __restrict__ c1,
    const uint64_t* __restrict__ eval_ax,
    const uint64_t* __restrict__ eval_bx,
    const size_t N,
    const size_t length,
    const size_t mult_length,
    const size_t beta,
    size_t curr_limbs,
    size_t alpha,
    size_t prime_gap,
    size_t special_mod_start,
    const uint64_t* __restrict__ primes,
    const uint64_t* __restrict__ barret_ks,
    const uint64_t* __restrict__ barret_ratios) {
  const int idx = blockIdx.y;
  const int coeff = blockIdx.x * blockDim.x + threadIdx.x;
  if (coeff >= N) {
    return;
  }

  const int swk_gap =
      static_cast<int>(special_mod_start) - static_cast<int>(curr_limbs);
  const int prime_idx = (idx < curr_limbs) ? 0 : static_cast<int>(prime_gap);
  const int swk_idx = (idx < curr_limbs) ? 0 : swk_gap;
  const int i = idx * N + coeff;

  const auto reduce_prime_idx = idx + prime_idx;
  const auto prime = primes[reduce_prime_idx];
  const auto barret_ratio = barret_ratios[reduce_prime_idx];
  const auto barret_k = barret_ks[reduce_prime_idx];

  uint128_t accum_ax = {0, 0};
  uint128_t accum_bx = {0, 0};
  for (int beta_idx = 0; beta_idx < beta; beta_idx++) {
    const int begin_idx = beta_idx * alpha;
    const int group_size =
        min(static_cast<int>(alpha), static_cast<int>(curr_limbs) - begin_idx);
    const bool is_original_limb =
        idx >= begin_idx && idx < begin_idx + group_size;
    const int stride = N * (mult_length * beta_idx + swk_idx);
    const int in_ptr_stride = N * length * beta_idx;
    const uint64_t op1 =
        is_original_limb ? c1[i] : in_modup[i + in_ptr_stride];
    const auto mul_ax = mult_64_64_128(op1, eval_ax[i + stride]);
    const auto mul_bx = mult_64_64_128(op1, eval_bx[i + stride]);
    inplace_add_128_128(mul_ax, accum_ax);
    inplace_add_128_128(mul_bx, accum_bx);
  }

  out_ax[i] = barret_reduction_128_64(accum_ax, prime, barret_ratio, barret_k);
  out_bx[i] = barret_reduction_128_64(accum_bx, prime, barret_ratio, barret_k);
}

__global__ void hrot_moddown_finalize_kernel(
    uint64_t* __restrict__ out_bx,
    uint64_t* __restrict__ out_ax,
    const uint64_t* __restrict__ baseconv_ntt,
    const uint64_t* __restrict__ key_products,
    const uint64_t* __restrict__ c0,
    const int* __restrict__ precomp_map,
    const uint64_t* __restrict__ prod_inv,
    const uint64_t* __restrict__ prod_inv_shoup,
    const uint64_t* __restrict__ primes,
    const int64_t N,
    const int64_t L_IN,
    const int64_t curr_limbs) {
  const int64_t j = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  const int64_t limb = blockIdx.y;
  if (j >= N || limb >= curr_limbs) {
    return;
  }

  const int64_t src = precomp_map[j];
  const int64_t q_index = limb * N + src;
  const int64_t out_index = limb * N + j;
  const int64_t cv_stride = L_IN * N;
  const uint64_t prime = primes[limb];
  const uint64_t inv = prod_inv[limb];
  const uint64_t inv_shoup = prod_inv_shoup[limb];

  uint64_t bx = sub_mod(key_products[q_index], baseconv_ntt[q_index], prime);
  bx = mul_and_reduce_shoup(bx, inv, inv_shoup, prime);
  if (bx >= prime) {
    bx -= prime;
  }
  out_bx[out_index] = add_mod(bx, c0[q_index], prime);

  uint64_t ax = sub_mod(
      key_products[cv_stride + q_index],
      baseconv_ntt[cv_stride + q_index],
      prime);
  ax = mul_and_reduce_shoup(ax, inv, inv_shoup, prime);
  if (ax >= prime) {
    ax -= prime;
  }
  out_ax[out_index] = ax;
}

} // namespace fhe

namespace at::native {

static Tensor hrot_innerproduct_cuda(
    const Tensor& in,
    const Tensor& c1,
    const Tensor& bx,
    const Tensor& ax,
    int64_t curr_limbs,
    int64_t alpha,
    int64_t special_mod_start,
    int64_t L,
    int64_t N,
    const Tensor& primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k) {
  TORCH_INTERNAL_ASSERT(in.dim() == 4);
  TORCH_INTERNAL_ASSERT(in.sizes()[0] == 1);
  TORCH_INTERNAL_ASSERT(in.sizes()[1] == 1);
  TORCH_INTERNAL_ASSERT(c1.dim() == 2);
  TORCH_INTERNAL_ASSERT(c1.sizes()[0] == curr_limbs);
  TORCH_INTERNAL_ASSERT(c1.sizes()[1] == N);
  const int beta = int((curr_limbs + alpha - 1) / alpha);
  int64_t sizeQP = primes.numel();
  int64_t sizeP = sizeQP - L;
  const int length = (curr_limbs + sizeP);
  const int mult_length = (special_mod_start + sizeP);
  TORCH_INTERNAL_ASSERT(in.sizes()[2] == beta * length);
  TORCH_INTERNAL_ASSERT(in.sizes()[3] == N);
  TORCH_CHECK(
      special_mod_start >= curr_limbs,
      "special_mod_start must be >= curr_limbs");
  TORCH_CHECK(special_mod_start <= L, "special_mod_start must be <= L");
  TORCH_CHECK(bx.dim() == 3, "bx must be [beta, mult_length, N]");
  TORCH_CHECK(ax.dim() == 3, "ax must be [beta, mult_length, N]");
  TORCH_CHECK(bx.sizes() == ax.sizes(), "bx and ax must have identical shapes");
  TORCH_CHECK(bx.size(0) >= beta, "bx/ax beta dimension mismatch");
  TORCH_CHECK(bx.size(1) >= mult_length, "bx/ax modulus dimension mismatch");
  TORCH_CHECK(bx.size(2) == N, "bx/ax last dimension must equal N");

  auto out = at::empty({2, 1, length, N}, in.options());
  auto in_ptr = reinterpret_cast<uint64_t*>(in.data_ptr<uint64_t>());
  auto c1_ptr = reinterpret_cast<uint64_t*>(c1.data_ptr<uint64_t>());
  auto ax_ptr = reinterpret_cast<uint64_t*>(ax.data_ptr<uint64_t>());
  auto bx_ptr = reinterpret_cast<uint64_t*>(bx.data_ptr<uint64_t>());
  auto out_bx_ptr = reinterpret_cast<uint64_t*>(out[0].data_ptr<uint64_t>());
  auto out_ax_ptr = reinterpret_cast<uint64_t*>(out[1].data_ptr<uint64_t>());
  auto primes_ptr = reinterpret_cast<uint64_t*>(primes.data_ptr<uint64_t>());
  auto barret_ratio_ptr =
      reinterpret_cast<uint64_t*>(barret_ratio.data_ptr<uint64_t>());
  auto barret_k_ptr =
      reinterpret_cast<uint64_t*>(barret_k.data_ptr<uint64_t>());
  auto gridDim = dim3(num_blocks(N), length);
  auto blockDim = BLOCK_SIZE;
  auto stream = at::cuda::getCurrentCUDAStream();

  fhe::hrot_sum_reduce_from_modup_kernel<<<gridDim, blockDim, 0, stream>>>(
      out_ax_ptr,
      out_bx_ptr,
      in_ptr,
      c1_ptr,
      ax_ptr,
      bx_ptr,
      N,
      length,
      mult_length,
      beta,
      curr_limbs,
      alpha,
      L - curr_limbs,
      special_mod_start,
      primes_ptr,
      barret_k_ptr,
      barret_ratio_ptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

template <size_t LOG_N>
static void launch_hrot_moddown_base_convert_ntt_phase1(
    uint64_t* workspace_ptr,
    int64_t L_IN,
    int64_t curr_limbs,
    int64_t sizeP,
    const uint64_t* prod_q_i_mod_q_j_moddown,
    const uint64_t* primes,
    const uint64_t* barret_ratio,
    const uint64_t* barret_k,
    const uint64_t* power_of_roots,
    const uint64_t* power_of_roots_shoup,
    cudaStream_t stream) {
  constexpr size_t N = size_t{1} << LOG_N;
  constexpr size_t NUM_GROUPS = 8;
  fhe::hrot_moddown_base_convert_ntt_phase1_kernel<LOG_N, NUM_GROUPS>
      <<<dim3(N / (NUM_GROUPS * 8) / 8, curr_limbs, 2),
         NUM_GROUPS * 8,
         0,
         stream>>>(
          workspace_ptr,
          L_IN,
          curr_limbs,
          sizeP,
          prod_q_i_mod_q_j_moddown,
          primes,
          barret_ratio,
          barret_k,
          power_of_roots,
          power_of_roots_shoup);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static void hrot_moddown_base_convert_ntt_phase1_cuda(
    uint64_t* workspace_ptr,
    int64_t L_IN,
    int64_t curr_limbs,
    int64_t sizeP,
    int64_t N,
    const Tensor& prod_q_i_mod_q_j_moddown,
    const Tensor& primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    const Tensor& power_of_roots,
    const Tensor& power_of_roots_shoup) {
  auto stream = at::cuda::getCurrentCUDAStream();
  if (N == (int64_t{1} << 17)) {
    launch_hrot_moddown_base_convert_ntt_phase1<17>(
        workspace_ptr,
        L_IN,
        curr_limbs,
        sizeP,
        prod_q_i_mod_q_j_moddown.data_ptr<uint64_t>(),
        primes.data_ptr<uint64_t>(),
        barret_ratio.data_ptr<uint64_t>(),
        barret_k.data_ptr<uint64_t>(),
        power_of_roots.data_ptr<uint64_t>(),
        power_of_roots_shoup.data_ptr<uint64_t>(),
        stream);
  } else if (N == (int64_t{1} << 16)) {
    launch_hrot_moddown_base_convert_ntt_phase1<16>(
        workspace_ptr,
        L_IN,
        curr_limbs,
        sizeP,
        prod_q_i_mod_q_j_moddown.data_ptr<uint64_t>(),
        primes.data_ptr<uint64_t>(),
        barret_ratio.data_ptr<uint64_t>(),
        barret_k.data_ptr<uint64_t>(),
        power_of_roots.data_ptr<uint64_t>(),
        power_of_roots_shoup.data_ptr<uint64_t>(),
        stream);
  } else if (N == (int64_t{1} << 15)) {
    launch_hrot_moddown_base_convert_ntt_phase1<15>(
        workspace_ptr,
        L_IN,
        curr_limbs,
        sizeP,
        prod_q_i_mod_q_j_moddown.data_ptr<uint64_t>(),
        primes.data_ptr<uint64_t>(),
        barret_ratio.data_ptr<uint64_t>(),
        barret_k.data_ptr<uint64_t>(),
        power_of_roots.data_ptr<uint64_t>(),
        power_of_roots_shoup.data_ptr<uint64_t>(),
        stream);
  } else if (N == (int64_t{1} << 14)) {
    launch_hrot_moddown_base_convert_ntt_phase1<14>(
        workspace_ptr,
        L_IN,
        curr_limbs,
        sizeP,
        prod_q_i_mod_q_j_moddown.data_ptr<uint64_t>(),
        primes.data_ptr<uint64_t>(),
        barret_ratio.data_ptr<uint64_t>(),
        barret_k.data_ptr<uint64_t>(),
        power_of_roots.data_ptr<uint64_t>(),
        power_of_roots_shoup.data_ptr<uint64_t>(),
        stream);
  } else {
    TORCH_INTERNAL_ASSERT(false, "Unsupported hrot NTT size");
  }
}

static std::vector<Tensor> hrot_moddown_cuda(
    const Tensor& in,
    const Tensor& c0,
    const Tensor& precomp_map,
    int64_t curr_limbs,
    int64_t L,
    int64_t sizeP,
    int64_t N,
    const Tensor& hat_inverse_vec_moddown,
    const Tensor& hat_inverse_vec_shoup_moddown,
    const Tensor& prod_q_i_mod_q_j_moddown,
    const Tensor& prod_inv_moddown,
    const Tensor& prod_inv_shoup_moddown,
    const Tensor& primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    const Tensor& power_of_roots_shoup,
    const Tensor& power_of_roots,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two) {
  TORCH_INTERNAL_ASSERT(in.dim() == 4);
  TORCH_INTERNAL_ASSERT(in.sizes()[0] == 2);
  TORCH_INTERNAL_ASSERT(in.sizes()[1] == 1);
  TORCH_INTERNAL_ASSERT(in.sizes()[2] == curr_limbs + sizeP);
  TORCH_INTERNAL_ASSERT(in.sizes()[3] == N);
  TORCH_INTERNAL_ASSERT(c0.dim() == 2);
  TORCH_INTERNAL_ASSERT(c0.sizes()[0] == curr_limbs);
  TORCH_INTERNAL_ASSERT(c0.sizes()[1] == N);
  TORCH_INTERNAL_ASSERT(precomp_map.dim() == 1);
  TORCH_INTERNAL_ASSERT(precomp_map.sizes()[0] == N);

  const int64_t num_cv = 2;
  const int64_t batch = 1;
  const int64_t L_IN = curr_limbs + sizeP;
  auto workspace = at::empty({num_cv, batch, L_IN, N}, in.options());
  auto out_bx = at::empty({curr_limbs, N}, c0.options());
  auto out_ax = at::empty({curr_limbs, N}, c0.options());

  auto from_ptr = reinterpret_cast<uint64_t*>(in.data_ptr<uint64_t>());
  auto workspace_ptr =
      reinterpret_cast<uint64_t*>(workspace.data_ptr<uint64_t>());

  iNTT_scaled_impl(
      workspace_ptr + curr_limbs * N,
      from_ptr + curr_limbs * N,
      sizeP,
      N,
      L_IN,
      L_IN,
      num_cv,
      batch,
      primes.data_ptr<uint64_t>() + L,
      inverse_power_of_roots_div_two.data_ptr<uint64_t>() + L * N,
      inverse_scaled_power_of_roots_div_two.data_ptr<uint64_t>() + L * N,
      hat_inverse_vec_moddown.data_ptr<uint64_t>(),
      hat_inverse_vec_shoup_moddown.data_ptr<uint64_t>());

  hrot_moddown_base_convert_ntt_phase1_cuda(
      workspace_ptr,
      L_IN,
      curr_limbs,
      sizeP,
      N,
      prod_q_i_mod_q_j_moddown,
      primes,
      barret_ratio,
      barret_k,
      power_of_roots,
      power_of_roots_shoup);

  NTT_phase2_impl(
      workspace_ptr,
      curr_limbs,
      N,
      L_IN,
      num_cv,
      batch,
      primes.data_ptr<uint64_t>(),
      power_of_roots_shoup.data_ptr<uint64_t>(),
      power_of_roots.data_ptr<uint64_t>());

  dim3 block(BLOCK_SIZE);
  auto stream = at::cuda::getCurrentCUDAStream();
  fhe::hrot_moddown_finalize_kernel<<<
      dim3(num_blocks(N), curr_limbs),
      block,
      0,
      stream>>>(
      out_bx.data_ptr<uint64_t>(),
      out_ax.data_ptr<uint64_t>(),
      workspace_ptr,
      from_ptr,
      c0.data_ptr<uint64_t>(),
      precomp_map.data_ptr<int>(),
      prod_inv_moddown.data_ptr<uint64_t>(),
      prod_inv_shoup_moddown.data_ptr<uint64_t>(),
      primes.data_ptr<uint64_t>(),
      N,
      L_IN,
      curr_limbs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out_bx, out_ax};
}

std::vector<Tensor> hrot_cuda(
    const Tensor& c0,
    const Tensor& c1,
    const Tensor& bx,
    const Tensor& ax,
    const Tensor& precomp_map,
    int64_t curr_limbs,
    int64_t special_mod_start,
    int64_t L,
    int64_t beta,
    int64_t N,
    int64_t alpha,
    const Tensor& hat_inverse_vec_modup,
    const Tensor& hat_inverse_vec_shoup_modup,
    const Tensor& prod_q_i_mod_q_j_modup,
    const Tensor& hat_inverse_vec_moddown,
    const Tensor& hat_inverse_vec_shoup_moddown,
    const Tensor& prod_q_i_mod_q_j_moddown,
    const Tensor& prod_inv_moddown,
    const Tensor& prod_inv_shoup_moddown,
    const Tensor& primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    const Tensor& power_of_roots_shoup,
    const Tensor& power_of_roots,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& inner_workspace) {
  TORCH_INTERNAL_ASSERT(c0.dim() == 2);
  TORCH_INTERNAL_ASSERT(c1.dim() == 2);
  TORCH_INTERNAL_ASSERT(c0.sizes()[0] == curr_limbs);
  TORCH_INTERNAL_ASSERT(c1.sizes()[0] == curr_limbs);
  TORCH_INTERNAL_ASSERT(c0.sizes()[1] == N);
  TORCH_INTERNAL_ASSERT(c1.sizes()[1] == N);
  TORCH_INTERNAL_ASSERT(precomp_map.dim() == 1);
  TORCH_INTERNAL_ASSERT(precomp_map.sizes()[0] == N);

  const auto sizeP = primes.numel() - L;
  const auto c1_4d = at::reshape(c1, {1, 1, curr_limbs, N});
  const auto modup = modup_without_copy_cuda(
      c1_4d,
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
  const auto inner_product = hrot_innerproduct_cuda(
      modup,
      c1,
      bx,
      ax,
      curr_limbs,
      alpha,
      special_mod_start,
      L,
      N,
      primes,
      barret_ratio,
      barret_k);
  return hrot_moddown_cuda(
      inner_product,
      c0,
      precomp_map,
      curr_limbs,
      L,
      sizeP,
      N,
      hat_inverse_vec_moddown,
      hat_inverse_vec_shoup_moddown,
      prod_q_i_mod_q_j_moddown,
      prod_inv_moddown,
      prod_inv_shoup_moddown,
      primes,
      barret_ratio,
      barret_k,
      power_of_roots_shoup,
      power_of_roots,
      inverse_power_of_roots_div_two,
      inverse_scaled_power_of_roots_div_two);
}

} // namespace at::native
