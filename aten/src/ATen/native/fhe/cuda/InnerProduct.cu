#include <ATen/Dispatch_v2.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/interned_strings.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/thread_constants.h>
#include <ATen/ops/copy.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/stack.h>
#include <ATen/ops/zeros.h>

#include <vector>

#include "ATen/native/fhe/cuda/Utils.cuh"

__constant__ const uint64_t*  c_eval_swks[64];

namespace fhe {

// template <size_t BATCH>
// __global__ void sum_reduce_fused(
//     uint64_t* out_ax,
//     uint64_t* out_bx,
//     const uint64_t* in_ptr,
//     const uint64_t* eval_ax,
//     const uint64_t* eval_bx,
//     const size_t N,
//     const size_t length,
//     const size_t mult_length,
//     const size_t beta,
//     size_t curr_limbs,
//     size_t gap,
//     const uint64_t* primes,
//     const uint64_t* barret_ks,
//     const uint64_t* barret_ratios) {
//   const int idx = blockIdx.y;
//   const int i = idx * N + blockIdx.x * blockDim.x + threadIdx.x;
//   const int prime_idx = ((idx >= 0 && idx < curr_limbs) ? 0 : gap);
//   uint128_t accum_ax[BATCH];
//   uint128_t accum_bx[BATCH];
//   for (int j = 0; j < BATCH; j++) {
//     accum_ax[j] = {0, 0};
//     accum_bx[j] = {0, 0};
//   }
//   for (int beta_idx = 0; beta_idx < beta; beta_idx++) {
//     const int stride = N * (mult_length * beta_idx + prime_idx);
//     const int in_ptr_stride = N * length * beta_idx;
//     const uint64_t op2_ax = eval_ax[i + stride];
//     const uint64_t op2_bx = eval_bx[i + stride];
//     for(int j = 0; j < BATCH; j++) {
//       const uint64_t op1 = in_ptr[i + in_ptr_stride + j * (N * length *
//       beta)]; const auto mul_ax = mult_64_64_128(op1, op2_ax); const auto
//       mul_bx = mult_64_64_128(op1, op2_bx); inplace_add_128_128(mul_ax,
//       accum_ax[j]); inplace_add_128_128(mul_bx, accum_bx[j]);
//     }
//   }
//   const auto reduce_prime_idx = idx + prime_idx;

//   const auto prime = primes[reduce_prime_idx];
//   const auto barret_ratio = barret_ratios[reduce_prime_idx];
//   const auto barret_k = barret_ks[reduce_prime_idx];
//   for(int j = 0; j < BATCH; j++) {
//     auto res_ax =
//       barret_reduction_128_64(accum_ax[j], prime, barret_ratio, barret_k);
//     auto res_bx =
//       barret_reduction_128_64(accum_bx[j], prime, barret_ratio, barret_k);
//     out_ax[j * (N * length) + i] = res_ax;
//     out_bx[j * (N * length) + i] = res_bx;
//   }
// }

__global__ void sum_reduce_fused_broadcast_key(
    uint64_t* out_ax,
    uint64_t* out_bx,
    const uint64_t* in_ptr,
    const uint64_t* eval_ax,
    const uint64_t* eval_bx,
    const size_t N,
    const size_t BATCH,
    const size_t length,
    const size_t mult_length,
    const size_t beta,
    size_t curr_limbs,
    size_t prime_gap,
    size_t special_mod_start,
    const uint64_t* primes,
    const uint64_t* barret_ks,
    const uint64_t* barret_ratios) {
  const int idx = blockIdx.y;
  const int i = idx * N + blockIdx.x * blockDim.x + threadIdx.x;
  const int swk_gap =
      static_cast<int>(special_mod_start) - static_cast<int>(curr_limbs);
  const int prime_idx =
      ((idx >= 0 && idx < curr_limbs) ? 0 : static_cast<int>(prime_gap));
  const int swk_idx = ((idx >= 0 && idx < curr_limbs) ? 0 : swk_gap);

  extern __shared__ uint64_t shared_mem[];
  auto eval_bx_shm = reinterpret_cast<uint64_t*>(shared_mem);
  auto eval_ax_shm =
      reinterpret_cast<uint64_t*>(shared_mem + BLOCK_SIZE * beta);

  const auto reduce_prime_idx = idx + prime_idx;
  const auto prime = primes[reduce_prime_idx];
  const auto barret_ratio = barret_ratios[reduce_prime_idx];
  const auto barret_k = barret_ks[reduce_prime_idx];

  for (int beta_idx = 0; beta_idx < beta; beta_idx++) {
    const int stride = N * (mult_length * beta_idx + swk_idx);
    eval_bx_shm[threadIdx.x + beta_idx * BLOCK_SIZE] = eval_bx[i + stride];
    eval_ax_shm[threadIdx.x + beta_idx * BLOCK_SIZE] = eval_ax[i + stride];
  }
  __syncthreads();
  for (int j = 0; j < BATCH; j++) {
    uint128_t accum_ax = {0, 0};
    uint128_t accum_bx = {0, 0};
    for (int beta_idx = 0; beta_idx < beta; beta_idx++) {
      const int in_ptr_stride = N * length * beta_idx;
      const uint64_t op1 = in_ptr[i + in_ptr_stride + j * (N * length * beta)];
      const auto mul_ax =
          mult_64_64_128(op1, eval_ax_shm[threadIdx.x + beta_idx * BLOCK_SIZE]);
      const auto mul_bx =
          mult_64_64_128(op1, eval_bx_shm[threadIdx.x + beta_idx * BLOCK_SIZE]);
      inplace_add_128_128(mul_ax, accum_ax);
      inplace_add_128_128(mul_bx, accum_bx);
    }
    auto res_ax =
        barret_reduction_128_64(accum_ax, prime, barret_ratio, barret_k);
    auto res_bx =
        barret_reduction_128_64(accum_bx, prime, barret_ratio, barret_k);
    out_ax[j * (N * length) + i] = res_ax;
    out_bx[j * (N * length) + i] = res_bx;
  }
}

#define SWK_PARAMS_1  uint64_t* swk0
#define SWK_PARAMS_2  SWK_PARAMS_1, uint64_t* swk1
#define SWK_PARAMS_3  SWK_PARAMS_2, uint64_t* swk2
#define SWK_PARAMS_4  SWK_PARAMS_3, uint64_t* swk3
#define SWK_PARAMS_5  SWK_PARAMS_4, uint64_t* swk4
#define SWK_PARAMS_6  SWK_PARAMS_5, uint64_t* swk5
#define SWK_PARAMS_7  SWK_PARAMS_6, uint64_t* swk6
#define SWK_PARAMS_8  SWK_PARAMS_7, uint64_t* swk7
#define SWK_PARAMS_9  SWK_PARAMS_8, uint64_t* swk8
#define SWK_PARAMS_10 SWK_PARAMS_9, uint64_t* swk9
#define SWK_PARAMS_11 SWK_PARAMS_10, uint64_t* swk10
#define SWK_PARAMS_12 SWK_PARAMS_11, uint64_t* swk11
#define SWK_PARAMS_13 SWK_PARAMS_12, uint64_t* swk12
#define SWK_PARAMS_14 SWK_PARAMS_13, uint64_t* swk13
#define SWK_PARAMS_15 SWK_PARAMS_14, uint64_t* swk14
#define SWK_PARAMS_16 SWK_PARAMS_15, uint64_t* swk15
#define SWK_PARAMS_17 SWK_PARAMS_16, uint64_t* swk16
#define SWK_PARAMS_18 SWK_PARAMS_17, uint64_t* swk17
#define SWK_PARAMS_19 SWK_PARAMS_18, uint64_t* swk18
#define SWK_PARAMS_20 SWK_PARAMS_19, uint64_t* swk19
#define SWK_PARAMS_21 SWK_PARAMS_20, uint64_t* swk20
#define SWK_PARAMS_22 SWK_PARAMS_21, uint64_t* swk21
#define SWK_PARAMS_23 SWK_PARAMS_22, uint64_t* swk22
#define SWK_PARAMS_24 SWK_PARAMS_23, uint64_t* swk23
#define SWK_PARAMS_25 SWK_PARAMS_24, uint64_t* swk24
#define SWK_PARAMS_26 SWK_PARAMS_25, uint64_t* swk25
#define SWK_PARAMS_27 SWK_PARAMS_26, uint64_t* swk26
#define SWK_PARAMS_28 SWK_PARAMS_27, uint64_t* swk27
#define SWK_PARAMS_29 SWK_PARAMS_28, uint64_t* swk28
#define SWK_PARAMS_30 SWK_PARAMS_29, uint64_t* swk29
#define SWK_PARAMS_31 SWK_PARAMS_30, uint64_t* swk30
#define SWK_PARAMS_32 SWK_PARAMS_31, uint64_t* swk31

#define SWK_LIST_1  swk0
#define SWK_LIST_2  SWK_LIST_1, swk1
#define SWK_LIST_3  SWK_LIST_2, swk2
#define SWK_LIST_4  SWK_LIST_3, swk3
#define SWK_LIST_5  SWK_LIST_4, swk4
#define SWK_LIST_6  SWK_LIST_5, swk5
#define SWK_LIST_7  SWK_LIST_6, swk6
#define SWK_LIST_8  SWK_LIST_7, swk7
#define SWK_LIST_9  SWK_LIST_8, swk8
#define SWK_LIST_10 SWK_LIST_9, swk9
#define SWK_LIST_11 SWK_LIST_10, swk10
#define SWK_LIST_12 SWK_LIST_11, swk11
#define SWK_LIST_13 SWK_LIST_12, swk12
#define SWK_LIST_14 SWK_LIST_13, swk13
#define SWK_LIST_15 SWK_LIST_14, swk14
#define SWK_LIST_16 SWK_LIST_15, swk15
#define SWK_LIST_17 SWK_LIST_16, swk16
#define SWK_LIST_18 SWK_LIST_17, swk17
#define SWK_LIST_19 SWK_LIST_18, swk18
#define SWK_LIST_20 SWK_LIST_19, swk19
#define SWK_LIST_21 SWK_LIST_20, swk20
#define SWK_LIST_22 SWK_LIST_21, swk21
#define SWK_LIST_23 SWK_LIST_22, swk22
#define SWK_LIST_24 SWK_LIST_23, swk23
#define SWK_LIST_25 SWK_LIST_24, swk24
#define SWK_LIST_26 SWK_LIST_25, swk25
#define SWK_LIST_27 SWK_LIST_26, swk26
#define SWK_LIST_28 SWK_LIST_27, swk27
#define SWK_LIST_29 SWK_LIST_28, swk28
#define SWK_LIST_30 SWK_LIST_29, swk29
#define SWK_LIST_31 SWK_LIST_30, swk30
#define SWK_LIST_32 SWK_LIST_31, swk31

#define SWK_PARM_1  swks[0].data_ptr<uint64_t>()
#define SWK_PARM_2  SWK_PARM_1, swks[1].data_ptr<uint64_t>()
#define SWK_PARM_3  SWK_PARM_2, swks[2].data_ptr<uint64_t>()
#define SWK_PARM_4  SWK_PARM_3, swks[3].data_ptr<uint64_t>()
#define SWK_PARM_5  SWK_PARM_4, swks[4].data_ptr<uint64_t>()
#define SWK_PARM_6  SWK_PARM_5, swks[5].data_ptr<uint64_t>()
#define SWK_PARM_7  SWK_PARM_6, swks[6].data_ptr<uint64_t>()
#define SWK_PARM_8  SWK_PARM_7, swks[7].data_ptr<uint64_t>()
#define SWK_PARM_9  SWK_PARM_8, swks[8].data_ptr<uint64_t>()
#define SWK_PARM_10 SWK_PARM_9, swks[9].data_ptr<uint64_t>()
#define SWK_PARM_11 SWK_PARM_10, swks[10].data_ptr<uint64_t>()
#define SWK_PARM_12 SWK_PARM_11, swks[11].data_ptr<uint64_t>()
#define SWK_PARM_13 SWK_PARM_12, swks[12].data_ptr<uint64_t>()
#define SWK_PARM_14 SWK_PARM_13, swks[13].data_ptr<uint64_t>()
#define SWK_PARM_15 SWK_PARM_14, swks[14].data_ptr<uint64_t>()
#define SWK_PARM_16 SWK_PARM_15, swks[15].data_ptr<uint64_t>()
#define SWK_PARM_17 SWK_PARM_16, swks[16].data_ptr<uint64_t>()
#define SWK_PARM_18 SWK_PARM_17, swks[17].data_ptr<uint64_t>()
#define SWK_PARM_19 SWK_PARM_18, swks[18].data_ptr<uint64_t>()
#define SWK_PARM_20 SWK_PARM_19, swks[19].data_ptr<uint64_t>()
#define SWK_PARM_21 SWK_PARM_20, swks[20].data_ptr<uint64_t>()
#define SWK_PARM_22 SWK_PARM_21, swks[21].data_ptr<uint64_t>()
#define SWK_PARM_23 SWK_PARM_22, swks[22].data_ptr<uint64_t>()
#define SWK_PARM_24 SWK_PARM_23, swks[23].data_ptr<uint64_t>()
#define SWK_PARM_25 SWK_PARM_24, swks[24].data_ptr<uint64_t>()
#define SWK_PARM_26 SWK_PARM_25, swks[25].data_ptr<uint64_t>()
#define SWK_PARM_27 SWK_PARM_26, swks[26].data_ptr<uint64_t>()
#define SWK_PARM_28 SWK_PARM_27, swks[27].data_ptr<uint64_t>()
#define SWK_PARM_29 SWK_PARM_28, swks[28].data_ptr<uint64_t>()
#define SWK_PARM_30 SWK_PARM_29, swks[29].data_ptr<uint64_t>()
#define SWK_PARM_31 SWK_PARM_30, swks[30].data_ptr<uint64_t>()
#define SWK_PARM_32 SWK_PARM_31, swks[31].data_ptr<uint64_t>()

#define DEFINE_COMPUTE(BATCH_ID)                                          \
  {                                                                       \
    uint128_t accum_ax = {0, 0};                                          \
    uint128_t accum_bx = {0, 0};                                          \
    const int SWK_OFF = swk_off_list[BATCH_ID];                           \
    const int64_t special_mod_start = special_mod_start_list[BATCH_ID];   \
    const int swk_gap = static_cast<int>(special_mod_start - static_cast<int64_t>(curr_limbs)); \
    const int prime_idx = ((idx >= 0 && idx < curr_limbs) ? 0 : static_cast<int>(prime_gap)); \
    const int reduce_prime_idx = idx + prime_idx;                         \
    const auto prime = primes[reduce_prime_idx];                          \
    const auto barret_ratio = barret_ratios[reduce_prime_idx];            \
    const auto barret_k = barret_ks[reduce_prime_idx];                    \
    const int mult_length = static_cast<int>(special_mod_start + sizeP);  \
    const int swk_idx = ((idx >= 0 && idx < curr_limbs) ? 0 : swk_gap);  \
    const uint64_t* __restrict__ eval_swk = swk##BATCH_ID;                \
    for (int beta_idx = 0; beta_idx < beta; beta_idx++) {                 \
      const int stride = N * (mult_length * beta_idx + swk_idx);          \
      auto op1 = shared_mem[threadIdx.x + beta_idx * BLOCK_SIZE];         \
      auto op2_bx = eval_swk[i + stride];                                 \
      auto op2_ax = eval_swk[i + stride + SWK_OFF];                       \
      auto mul_ax = mult_64_64_128(op1, op2_ax);                          \
      auto mul_bx = mult_64_64_128(op1, op2_bx);                          \
      inplace_add_128_128(mul_ax, accum_ax);                              \
      inplace_add_128_128(mul_bx, accum_bx);                              \
    }                                                                     \
    auto res_ax =                                                         \
        barret_reduction_128_64(accum_ax, prime, barret_ratio, barret_k); \
    auto res_bx =                                                         \
        barret_reduction_128_64(accum_bx, prime, barret_ratio, barret_k); \
    out_ptr[BATCH_ID * (N * length) * 2 + i] = res_bx;                    \
    out_ptr[BATCH_ID * (N * length) * 2 + i + (N * length)] = res_ax;     \
  }

#define COMPUTE_ITER_1  DEFINE_COMPUTE(0)
#define COMPUTE_ITER_2  COMPUTE_ITER_1 DEFINE_COMPUTE(1)
#define COMPUTE_ITER_3  COMPUTE_ITER_2 DEFINE_COMPUTE(2)
#define COMPUTE_ITER_4  COMPUTE_ITER_3 DEFINE_COMPUTE(3)
#define COMPUTE_ITER_5  COMPUTE_ITER_4 DEFINE_COMPUTE(4)
#define COMPUTE_ITER_6  COMPUTE_ITER_5 DEFINE_COMPUTE(5)
#define COMPUTE_ITER_7  COMPUTE_ITER_6 DEFINE_COMPUTE(6)
#define COMPUTE_ITER_8  COMPUTE_ITER_7 DEFINE_COMPUTE(7)
#define COMPUTE_ITER_9  COMPUTE_ITER_8 DEFINE_COMPUTE(8)
#define COMPUTE_ITER_10 COMPUTE_ITER_9 DEFINE_COMPUTE(9)
#define COMPUTE_ITER_11 COMPUTE_ITER_10 DEFINE_COMPUTE(10)
#define COMPUTE_ITER_12 COMPUTE_ITER_11 DEFINE_COMPUTE(11)
#define COMPUTE_ITER_13 COMPUTE_ITER_12 DEFINE_COMPUTE(12)
#define COMPUTE_ITER_14 COMPUTE_ITER_13 DEFINE_COMPUTE(13)
#define COMPUTE_ITER_15 COMPUTE_ITER_14 DEFINE_COMPUTE(14)
#define COMPUTE_ITER_16 COMPUTE_ITER_15 DEFINE_COMPUTE(15)
#define COMPUTE_ITER_17 COMPUTE_ITER_16 DEFINE_COMPUTE(16)
#define COMPUTE_ITER_18 COMPUTE_ITER_17 DEFINE_COMPUTE(17)
#define COMPUTE_ITER_19 COMPUTE_ITER_18 DEFINE_COMPUTE(18)
#define COMPUTE_ITER_20 COMPUTE_ITER_19 DEFINE_COMPUTE(19)
#define COMPUTE_ITER_21 COMPUTE_ITER_20 DEFINE_COMPUTE(20)
#define COMPUTE_ITER_22 COMPUTE_ITER_21 DEFINE_COMPUTE(21)
#define COMPUTE_ITER_23 COMPUTE_ITER_22 DEFINE_COMPUTE(22)
#define COMPUTE_ITER_24 COMPUTE_ITER_23 DEFINE_COMPUTE(23)
#define COMPUTE_ITER_25 COMPUTE_ITER_24 DEFINE_COMPUTE(24)
#define COMPUTE_ITER_26 COMPUTE_ITER_25 DEFINE_COMPUTE(25)
#define COMPUTE_ITER_27 COMPUTE_ITER_26 DEFINE_COMPUTE(26)
#define COMPUTE_ITER_28 COMPUTE_ITER_27 DEFINE_COMPUTE(27)
#define COMPUTE_ITER_29 COMPUTE_ITER_28 DEFINE_COMPUTE(28)
#define COMPUTE_ITER_30 COMPUTE_ITER_29 DEFINE_COMPUTE(29)
#define COMPUTE_ITER_31 COMPUTE_ITER_30 DEFINE_COMPUTE(30)
#define COMPUTE_ITER_32 COMPUTE_ITER_31 DEFINE_COMPUTE(31)

#define DEFINE_KERNEL(BATCH)                                                \
  __global__ void sum_reduce_fused_broadcast_cipher_DL(                     \
      uint64_t* __restrict__ out_ptr,                                       \
      const uint64_t* __restrict__ in_ptr,                                  \
      SWK_PARAMS_##BATCH,                                                   \
      const int* __restrict__ swk_off_list,                                 \
      const size_t N,                                                       \
      const size_t length,                                                  \
      const size_t sizeP,                                                   \
      const size_t beta,                                                    \
      size_t curr_limbs,                                                    \
      size_t prime_gap,                                                     \
      const int64_t* special_mod_start_list,                                \
      const uint64_t* primes,                                               \
      const uint64_t* barret_ks,                                            \
      const uint64_t* barret_ratios) {                                      \
    const int idx = blockIdx.y;                                             \
    const int i = idx * N + blockIdx.x * blockDim.x + threadIdx.x;          \
    extern __shared__ uint64_t shared_mem[];                                \
    for (int beta_idx = 0; beta_idx < beta; beta_idx++) {                   \
      const int in_ptr_stride = N * length * beta_idx;                      \
      shared_mem[threadIdx.x + beta_idx * BLOCK_SIZE] =                     \
          in_ptr[i + in_ptr_stride];                                        \
    }                                                                       \
    __syncthreads();                                                        \
    COMPUTE_ITER_##BATCH;                                                   \
  }


DEFINE_KERNEL(1);
DEFINE_KERNEL(2);
DEFINE_KERNEL(3);
DEFINE_KERNEL(4);
DEFINE_KERNEL(5);
DEFINE_KERNEL(6);
DEFINE_KERNEL(7);
DEFINE_KERNEL(8);
DEFINE_KERNEL(9);
DEFINE_KERNEL(10);
DEFINE_KERNEL(11);
DEFINE_KERNEL(12);
DEFINE_KERNEL(13);
DEFINE_KERNEL(14);
DEFINE_KERNEL(15);
DEFINE_KERNEL(16);
DEFINE_KERNEL(17);
DEFINE_KERNEL(18);
DEFINE_KERNEL(19);
DEFINE_KERNEL(20);
DEFINE_KERNEL(21);
DEFINE_KERNEL(22);
DEFINE_KERNEL(23);
DEFINE_KERNEL(24);
DEFINE_KERNEL(25);
DEFINE_KERNEL(26);
DEFINE_KERNEL(27);
DEFINE_KERNEL(28);
DEFINE_KERNEL(29);
DEFINE_KERNEL(30);
DEFINE_KERNEL(31);
DEFINE_KERNEL(32);

} // namespace fhe

namespace at::native {

#define DISPATCH_SUM_REDUCE_CASE(NUM)            \
  case NUM:                                      \
    fhe::sum_reduce_fused_broadcast_cipher_DL<<< \
        gridDim,                                 \
        blockDim,                                \
        BLOCK_SIZE * beta * sizeof(uint64_t),    \
        stream>>>(                               \
        out_ptr,                                 \
        in_ptr,                                  \
        SWK_PARM_##NUM,                          \
        swk_off_ptr,                             \
        N,                                       \
        length,                                  \
        sizeP,                                   \
        beta,                                    \
        curr_limbs,                              \
        prime_gap,                               \
        special_mod_start_ptr,                   \
        primes_ptr,                              \
        barret_k_ptr,                            \
        barret_ratio_ptr);                       \
    break;

static void innerproduct_broadcast_cipher_template(
    Tensor& out,
    const Tensor& in,
    const TensorList& swks,
    int64_t curr_limbs,
    int64_t alpha,
    int64_t L,
    int64_t N,
    const Tensor& special_mod_start,
    const Tensor& primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    const Tensor& workspace) {
  const int beta = int((curr_limbs + alpha - 1) / alpha);
  int64_t sizeQP = primes.numel();
  int64_t sizeP = sizeQP - L;
  const int length = (curr_limbs + sizeP);
  const int prime_gap = L - curr_limbs;

  auto in_ptr = reinterpret_cast<uint64_t*>(in.data_ptr<uint64_t>());
  auto out_ptr = reinterpret_cast<uint64_t*>(out.data_ptr<uint64_t>());
  // auto SWK_OFF = swks[0].sizes()[1] * swks[0].sizes()[2] * swks[0].sizes()[3];
  const int B = static_cast<int>(swks.size());
  TORCH_INTERNAL_ASSERT(B > 0, "swks must not be empty");
  TORCH_INTERNAL_ASSERT(in.sizes()[0] == 1, "innerproduct_broadcast_cipher expects num_cv == 1");
  TORCH_INTERNAL_ASSERT(in.sizes()[1] == 1, "innerproduct_broadcast_cipher expects batch == 1");
  TORCH_INTERNAL_ASSERT(
      special_mod_start.numel() == B,
      "special_mod_start numel must equal swks.size()");
  TORCH_INTERNAL_ASSERT(
      special_mod_start.device() == in.device(),
      "special_mod_start must be on same device as in");

  at::Tensor special_mod_start_cpu = special_mod_start.cpu();
  const int64_t* special_mod_start_host = special_mod_start_cpu.data_ptr<int64_t>();
  at::Tensor swk_off_cpu = at::empty({B}, at::TensorOptions().dtype(at::kInt).device(at::kCPU));

  {
    int* swk_off_host = swk_off_cpu.data_ptr<int>();
    for (int b = 0; b < B; ++b) {
      TORCH_CHECK(
          special_mod_start_host[b] >= curr_limbs,
          "special_mod_start[", b, "] must be >= curr_limbs");
      TORCH_CHECK(
          special_mod_start_host[b] <= L,
          "special_mod_start[", b, "] must be <= L");
      TORCH_CHECK(swks[b].dim() == 4, "swk tensor must have shape [2, beta, mult_length, N]");
      const auto sizes = swks[b].sizes();
      TORCH_CHECK(sizes[0] == 2, "swk first dimension must be 2 (bx/ax)");
      TORCH_CHECK(sizes[1] >= beta, "swk beta dimension must be >= ceil(curr_limbs / alpha)");
      TORCH_CHECK(sizes[3] == N, "swk last dimension must equal N");
      const int64_t expected_mult_length = special_mod_start_host[b] + sizeP;
      TORCH_CHECK(
          sizes[2] >= expected_mult_length,
          "swk modulus dimension must be >= special_mod_start[b] + sizeP");
      const int off = static_cast<int>(sizes[1] * sizes[2] * sizes[3]);
      swk_off_host[b] = off;
    }
  }
  // 拷到 GPU（使用当前流，可 non_blocking）
  at::Tensor swk_off_dev = swk_off_cpu.to(in.device(), /*non_blocking=*/true);
  const int* swk_off_ptr = swk_off_dev.data_ptr<int>();

  auto special_mod_start_ptr = reinterpret_cast<int64_t*>(special_mod_start.data_ptr<int64_t>());
  auto primes_ptr = reinterpret_cast<uint64_t*>(primes.data_ptr<uint64_t>());
  auto barret_ratio_ptr =
      reinterpret_cast<uint64_t*>(barret_ratio.data_ptr<uint64_t>());
  auto barret_k_ptr =
      reinterpret_cast<uint64_t*>(barret_k.data_ptr<uint64_t>());
  auto gridDim = dim3(N / BLOCK_SIZE, length);
  auto blockDim = BLOCK_SIZE;
  auto stream = at::cuda::getCurrentCUDAStream();
  DISPATCH_BATCH_FUNC(swks.size(), DISPATCH_SUM_REDUCE_CASE);
}

Tensor innerproduct_broadcast_cipher_cuda(
    const Tensor& in,
    TensorList swks,
    int64_t curr_limbs,
    int64_t alpha,
    int64_t L,
    int64_t N,
    const Tensor& special_mod_start,
    const Tensor& primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    const Tensor& workspace) {
  TORCH_INTERNAL_ASSERT(in.dim() == 4);
  TORCH_INTERNAL_ASSERT(in.sizes()[0] == 1, "innerproduct_broadcast_cipher expects num_cv == 1");
  TORCH_INTERNAL_ASSERT(in.sizes()[1] == 1, "innerproduct_broadcast_cipher expects batch == 1");
  int batch = swks.size();

  int64_t sizeQP = primes.numel();
  int64_t sizeP = sizeQP - L;
  auto out = at::empty({batch, 2, curr_limbs + sizeP, N}, in.options());

  innerproduct_broadcast_cipher_template(
      out,
      in,
      swks,
      curr_limbs,
      alpha,
      L,
      N,
      special_mod_start,
      primes,
      barret_ratio,
      barret_k,
      workspace);
  return out;
}

static void innerproduct_template(
    Tensor& out,
    const Tensor& in,
    const Tensor& bx,
    const Tensor& ax,
    int64_t curr_limbs,
    int64_t alpha,
    int64_t special_mod_start,
    int64_t L,
    int64_t N,
    const Tensor& primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    const Tensor& workspace) {
  const int beta = int((curr_limbs + alpha - 1) / alpha);
  int64_t sizeQP = primes.numel();
  int64_t sizeP = sizeQP - L;
  const int length = (curr_limbs + sizeP);
  const int mult_length = (special_mod_start + sizeP);
  TORCH_CHECK(special_mod_start >= curr_limbs, "special_mod_start must be >= curr_limbs");
  TORCH_CHECK(special_mod_start <= L, "special_mod_start must be <= L");
  TORCH_CHECK(bx.dim() == 3, "bx must be [beta, mult_length, N]");
  TORCH_CHECK(ax.dim() == 3, "ax must be [beta, mult_length, N]");
  TORCH_CHECK(bx.sizes() == ax.sizes(), "bx and ax must have identical shapes");
  TORCH_CHECK(
      bx.size(0) >= beta,
      "bx/ax beta dimension must be >= ceil(curr_limbs / alpha)");
  TORCH_CHECK(
      bx.size(1) >= mult_length,
      "bx/ax modulus dimension must be >= special_mod_start + sizeP");
  TORCH_CHECK(bx.size(2) == N, "bx/ax last dimension must equal N");
  const int prime_gap = L - curr_limbs;

  auto in_ptr = reinterpret_cast<uint64_t*>(in.data_ptr<uint64_t>());
  auto ax_ptr = reinterpret_cast<uint64_t*>(ax.data_ptr<uint64_t>());
  auto bx_ptr = reinterpret_cast<uint64_t*>(bx.data_ptr<uint64_t>());
  auto out_bx_ptr = reinterpret_cast<uint64_t*>(out[0].data_ptr<uint64_t>());
  auto out_ax_ptr = reinterpret_cast<uint64_t*>(out[1].data_ptr<uint64_t>());
  auto primes_ptr = reinterpret_cast<uint64_t*>(primes.data_ptr<uint64_t>());
  auto barret_ratio_ptr =
      reinterpret_cast<uint64_t*>(barret_ratio.data_ptr<uint64_t>());
  auto barret_k_ptr =
      reinterpret_cast<uint64_t*>(barret_k.data_ptr<uint64_t>());
  auto gridDim = dim3(N / BLOCK_SIZE, length);
  auto blockDim = BLOCK_SIZE;
  auto stream = at::cuda::getCurrentCUDAStream();

  cudaFuncSetAttribute(
      fhe::sum_reduce_fused_broadcast_key,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      BLOCK_SIZE * beta * sizeof(uint64_t) * 2);

  fhe::sum_reduce_fused_broadcast_key<<<
      gridDim,
      blockDim,
      BLOCK_SIZE * beta * sizeof(uint64_t) * 2,
      stream>>>(
      out_ax_ptr,
      out_bx_ptr,
      in_ptr,
      ax_ptr,
      bx_ptr,
      N,
      out.sizes()[1],
      length,
      mult_length,
      beta,
      curr_limbs,
      prime_gap,
      special_mod_start,
      primes_ptr,
      barret_k_ptr,
      barret_ratio_ptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static void innerproduct_write_pair_template(
    Tensor& out_bx,
    Tensor& out_ax,
    const Tensor& in,
    const Tensor& bx,
    const Tensor& ax,
    int64_t curr_limbs,
    int64_t alpha,
    int64_t special_mod_start,
    int64_t L,
    int64_t N,
    const Tensor& primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    const Tensor& workspace) {
  const int beta = int((curr_limbs + alpha - 1) / alpha);
  int64_t sizeQP = primes.numel();
  int64_t sizeP = sizeQP - L;
  const int length = (curr_limbs + sizeP);
  const int mult_length = (special_mod_start + sizeP);
  TORCH_CHECK(special_mod_start >= curr_limbs, "special_mod_start must be >= curr_limbs");
  TORCH_CHECK(special_mod_start <= L, "special_mod_start must be <= L");
  TORCH_CHECK(out_bx.dim() == 3, "out_bx must be [batch, length, N]");
  TORCH_CHECK(out_ax.dim() == 3, "out_ax must be [batch, length, N]");
  TORCH_CHECK(out_bx.sizes() == out_ax.sizes(), "out_bx and out_ax must have identical shapes");
  TORCH_CHECK(out_bx.size(1) == length, "innerproduct_write_pair output limb dimension mismatch");
  TORCH_CHECK(out_bx.size(2) == N, "innerproduct_write_pair output N mismatch");
  TORCH_CHECK(bx.dim() == 3, "bx must be [beta, mult_length, N]");
  TORCH_CHECK(ax.dim() == 3, "ax must be [beta, mult_length, N]");
  TORCH_CHECK(bx.sizes() == ax.sizes(), "bx and ax must have identical shapes");
  TORCH_CHECK(
      bx.size(0) >= beta,
      "bx/ax beta dimension must be >= ceil(curr_limbs / alpha)");
  TORCH_CHECK(
      bx.size(1) >= mult_length,
      "bx/ax modulus dimension must be >= special_mod_start + sizeP");
  TORCH_CHECK(bx.size(2) == N, "bx/ax last dimension must equal N");
  const int prime_gap = L - curr_limbs;

  auto in_ptr = reinterpret_cast<uint64_t*>(in.data_ptr<uint64_t>());
  auto ax_ptr = reinterpret_cast<uint64_t*>(ax.data_ptr<uint64_t>());
  auto bx_ptr = reinterpret_cast<uint64_t*>(bx.data_ptr<uint64_t>());
  auto out_bx_ptr = reinterpret_cast<uint64_t*>(out_bx.data_ptr<uint64_t>());
  auto out_ax_ptr = reinterpret_cast<uint64_t*>(out_ax.data_ptr<uint64_t>());
  auto primes_ptr = reinterpret_cast<uint64_t*>(primes.data_ptr<uint64_t>());
  auto barret_ratio_ptr =
      reinterpret_cast<uint64_t*>(barret_ratio.data_ptr<uint64_t>());
  auto barret_k_ptr =
      reinterpret_cast<uint64_t*>(barret_k.data_ptr<uint64_t>());
  auto gridDim = dim3(N / BLOCK_SIZE, length);
  auto blockDim = BLOCK_SIZE;
  auto stream = at::cuda::getCurrentCUDAStream();

  cudaFuncSetAttribute(
      fhe::sum_reduce_fused_broadcast_key,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      BLOCK_SIZE * beta * sizeof(uint64_t) * 2);

  fhe::sum_reduce_fused_broadcast_key<<<
      gridDim,
      blockDim,
      BLOCK_SIZE * beta * sizeof(uint64_t) * 2,
      stream>>>(
      out_ax_ptr,
      out_bx_ptr,
      in_ptr,
      ax_ptr,
      bx_ptr,
      N,
      out_bx.sizes()[0],
      length,
      mult_length,
      beta,
      curr_limbs,
      prime_gap,
      special_mod_start,
      primes_ptr,
      barret_k_ptr,
      barret_ratio_ptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

Tensor innerproduct_cuda(
    const Tensor& in,
    const Tensor& bx,
    const Tensor& ax,
    int64_t curr_limbs,
    int64_t alpha,
    int64_t special_mod_start,
    int64_t L,
    int64_t N,
    const Tensor& primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    const Tensor& workspace) {
  TORCH_INTERNAL_ASSERT(in.dim() == 4);
  auto num_cv = in.sizes()[0]; // should be 1
  auto batch = in.sizes()[1];

  int64_t sizeQP = primes.numel();
  int64_t sizeP = sizeQP - L;
  auto out = at::empty({2, batch, curr_limbs + sizeP, N}, in.options());

  innerproduct_template(
      out,
      in,
      bx,
      ax,
      curr_limbs,
      alpha,
      special_mod_start,
      L,
      N,
      primes,
      barret_ratio,
      barret_k,
      workspace);
  return out;
}

Tensor innerproduct_write_cuda(
    const Tensor& out,
    const Tensor& in,
    const Tensor& bx,
    const Tensor& ax,
    int64_t curr_limbs,
    int64_t alpha,
    int64_t special_mod_start,
    int64_t L,
    int64_t N,
    const Tensor& primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    const Tensor& workspace) {
  TORCH_INTERNAL_ASSERT(in.dim() == 4);
  TORCH_INTERNAL_ASSERT(out.dim() == 4);
  TORCH_INTERNAL_ASSERT(out.sizes()[0] == 2, "innerproduct_write expects out num_cv == 2");
  int64_t sizeQP = primes.numel();
  int64_t sizeP = sizeQP - L;
  TORCH_INTERNAL_ASSERT(
      out.sizes()[2] == curr_limbs + sizeP,
      "innerproduct_write output limb dimension mismatch");
  TORCH_INTERNAL_ASSERT(out.sizes()[3] == N, "innerproduct_write output N mismatch");

  auto mutable_out = out;
  innerproduct_template(
      mutable_out,
      in,
      bx,
      ax,
      curr_limbs,
      alpha,
      special_mod_start,
      L,
      N,
      primes,
      barret_ratio,
      barret_k,
      workspace);
  return out;
}

std::vector<Tensor> innerproduct_write_pair_cuda(
    const Tensor& out_bx,
    const Tensor& out_ax,
    const Tensor& in,
    const Tensor& bx,
    const Tensor& ax,
    int64_t curr_limbs,
    int64_t alpha,
    int64_t special_mod_start,
    int64_t L,
    int64_t N,
    const Tensor& primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    const Tensor& workspace) {
  TORCH_INTERNAL_ASSERT(in.dim() == 4);

  auto mutable_out_bx = out_bx;
  auto mutable_out_ax = out_ax;
  innerproduct_write_pair_template(
      mutable_out_bx,
      mutable_out_ax,
      in,
      bx,
      ax,
      curr_limbs,
      alpha,
      special_mod_start,
      L,
      N,
      primes,
      barret_ratio,
      barret_k,
      workspace);
  return {out_bx, out_ax};
}

} // namespace at::native
