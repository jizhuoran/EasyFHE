#include <ATen/cuda/CUDAContext.h>

#include "ATen/native/fhe/cuda/device/Ntt.cuh"

namespace fhe {
__device__ __forceinline__ void butt_intt_local(
    uint64_t& x,
    uint64_t& y,
    const uint64_t w,
    const uint64_t w_,
    const uint64_t p,
    const uint64_t two_p) {
  const uint64_t T = two_p - y + x;
  uint64_t new_x = x + y;
  if (new_x >= two_p)
    new_x -= two_p;
  if (T & 1)
    new_x += p;
  x = (new_x >> 1);
  y = mul_and_reduce_shoup(T, w, w_, p);
}

template <int NUM_ROUNDS>
__device__ __forceinline__ void intt_warp_butterfly(
    uint64_t& i1,
    uint64_t& i2,
    uint32_t& stage_off,
    const uint32_t localID,
    const uint64_t* __restrict__ base_inv,
    const uint64_t* __restrict__ base_inv_,
    const uint64_t prime,
    const uint64_t two_p) {
  static_assert(NUM_ROUNDS >= 1);
  butt_intt_local(
      i1,
      i2,
      base_inv[stage_off + localID],
      base_inv_[stage_off + localID],
      prime,
      two_p);

#pragma unroll
  for (int shift = 1; shift < NUM_ROUNDS;
       ++shift) { // offsets: 2, 4, 8, 16, ...
    const uint32_t offset = 1u << shift; // 2^shift
    const bool lower_half = (localID & (offset - 1)) < (offset >> 1);

    // choose the value to exchange, then shuffle across the offset‑distance
    auto tmp = lower_half ? i2 : i1;
    tmp = __shfl_xor_sync(0xFFFFFFFF, tmp, offset >> 1);

    lower_half ? i2 = tmp : i1 = tmp; // exchange values

    // advance table pointer for the current NTT stage
    stage_off >>= 1; // equivalent to stage_off /= 2
    const uint32_t idx = stage_off + (localID >> shift);

    butt_intt_local(i1, i2, base_inv[idx], base_inv_[idx], prime, two_p);
  }
}

template <size_t LOG_N>
__global__ void INTTXPointPhase1(
    const uint64_t* __restrict__ in_ptr,
    uint64_t* __restrict__ out_ptr,
    const size_t LOG_CV,
    const size_t L_OUTN,
    const size_t BL_OUTN,
    const size_t L_INN,
    const size_t BL_INN,
    const uint64_t* __restrict__ base_inv,
    const uint64_t* __restrict__ base_inv_,
    const uint64_t* __restrict__ primes) {
  constexpr size_t LOG_RADIX = LOG_N - 6;
  static_assert(LOG_RADIX >= 8);
  const int R2_RADIX = 64;
  const int DATA_SIZE = 1u << LOG_RADIX; // 1024 for 2^10, 2048 for 2^11
  const int kBLOCK_SIZE = 1u << (LOG_RADIX - 1);

  auto cipher_id = blockIdx.z >> LOG_CV;
  auto cv_id = blockIdx.z & ((1 << LOG_CV) - 1);
  in_ptr += cv_id * BL_INN + cipher_id * L_INN;
  out_ptr += cv_id * BL_OUTN + cipher_id * L_OUTN;

  const uint32_t batch_idx = blockIdx.y;
  base_inv += batch_idx * (1u << LOG_N);
  base_inv_ += batch_idx * (1u << LOG_N);
  const uint64_t prime = primes[batch_idx];
  const uint64_t two_p = prime << 1;

  auto in_row =
      reinterpret_cast<const ulonglong2(*)[R2_RADIX][DATA_SIZE / 2]>(in_ptr);
  ulonglong2 i12 = in_row[batch_idx][blockIdx.x][threadIdx.x];
  uint64_t i1 = i12.x;
  uint64_t i2 = i12.y;

  uint32_t stage_off = (1u << (LOG_N - 1)) + blockIdx.x * blockDim.x;
  intt_warp_butterfly<6>(
      i1,
      i2,
      stage_off,
      threadIdx.x,
      base_inv,
      base_inv_,
      prime,
      two_p);

  uint32_t localID = threadIdx.x % WARP_SIZE;
  uint32_t groupID = threadIdx.x / WARP_SIZE;
  const int NUM_WARP = kBLOCK_SIZE / WARP_SIZE;
  __shared__ uint64_t tile[2][NUM_WARP][WARP_SIZE + 1];
  tile[0][groupID][localID] = i1;
  tile[1][groupID][localID] = i2;
  __syncthreads();
  auto SECOND_GROUP_SIZE = DATA_SIZE / WARP_SIZE / 4; // 8 for 2^10, 16 for 2^11

  localID = threadIdx.x / SECOND_GROUP_SIZE; // 0 ~ 63
  groupID = threadIdx.x % SECOND_GROUP_SIZE; // 0 ~ 7 for 2^10, 0 ~ 15 for 2^11

  const uint32_t read_plane = localID >> 5;
  const uint32_t read_col = localID & (WARP_SIZE - 1);
  i1 = tile[read_plane][2 * groupID][read_col];
  i2 = tile[read_plane][2 * groupID + 1][read_col];

  stage_off = stage_off >> 1; // equivalent to stage_off /= 2
  intt_warp_butterfly<LOG_RADIX - 6>(
      i1,
      i2,
      stage_off,
      groupID, // laneID
      base_inv,
      base_inv_,
      prime,
      two_p);

  tile[read_plane][groupID][read_col] = i1;
  tile[read_plane][groupID + SECOND_GROUP_SIZE][read_col] = i2;
  __syncthreads();

  auto out_row = reinterpret_cast<uint64_t(*)[R2_RADIX][DATA_SIZE]>(out_ptr);
  const uint32_t out_col = threadIdx.x % (2 * WARP_SIZE);
  const uint32_t out_plane = out_col >> 5;
  const uint32_t out_lane = out_col & (WARP_SIZE - 1);
  out_row[batch_idx][blockIdx.x][threadIdx.x] =
      tile[out_plane][threadIdx.x / (2 * WARP_SIZE)][out_lane];
  out_row[batch_idx][blockIdx.x][threadIdx.x + kBLOCK_SIZE] =
      tile[out_plane][(threadIdx.x + kBLOCK_SIZE) / (2 * WARP_SIZE)]
          [out_lane];
}

template <uint8_t radix>
__device__ __forceinline__ void LOCAL_INTT_RADIX(
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
    const uint64_t* __restrict__ W_,
    const uint64_t prime,
    const uint64_t two_p) {
  static_assert(radix == 2 || radix == 4 || radix == 8);
  if constexpr (radix >= 8) {
    const uint32_t off = 4 * tw_off;
    const uint64_t w0 = W[off];
    const uint64_t ws0 = W_[off];
    const uint64_t w1 = W[off + 1];
    const uint64_t ws1 = W_[off + 1];
    const uint64_t w2 = W[off + 2];
    const uint64_t ws2 = W_[off + 2];
    const uint64_t w3 = W[off + 3];
    const uint64_t ws3 = W_[off + 3];
    butt_intt_local(local0, local1, w0, ws0, prime, two_p);
    butt_intt_local(local2, local3, w1, ws1, prime, two_p);
    butt_intt_local(local4, local5, w2, ws2, prime, two_p);
    butt_intt_local(local6, local7, w3, ws3, prime, two_p);
  }

  if constexpr (radix >= 4) {
    const uint32_t off = 2 * tw_off;
    const uint64_t w0 = W[off];
    const uint64_t ws0 = W_[off];
    const uint64_t w1 = W[off + 1];
    const uint64_t ws1 = W_[off + 1];
    butt_intt_local(local0, local2, w0, ws0, prime, two_p);
    butt_intt_local(local1, local3, w0, ws0, prime, two_p);
    butt_intt_local(local4, local6, w1, ws1, prime, two_p);
    butt_intt_local(local5, local7, w1, ws1, prime, two_p);
  }
  if constexpr (radix >= 2) {
    const uint64_t w = W[tw_off];
    const uint64_t ws = W_[tw_off];
    butt_intt_local(local0, local4, w, ws, prime, two_p);
    butt_intt_local(local1, local5, w, ws, prime, two_p);
    butt_intt_local(local2, local6, w, ws, prime, two_p);
    butt_intt_local(local3, local7, w, ws, prime, two_p);
  }
}

template <size_t LOG_N, size_t NUM_GROUPS>
__global__ void INTT64PointPhase2(
    uint64_t __restrict__* inout_ptr,
    const size_t LOG_CV,
    const size_t LN,
    const size_t BLN,
    const uint64_t __restrict__* base_inv,
    const uint64_t __restrict__* base_inv_,
    const uint64_t __restrict__* primes) {
  static_assert(NUM_GROUPS == 8);
  const int GROUP_SIZE = 8;

  auto cipher_id = blockIdx.z >> LOG_CV;
  auto cv_id = blockIdx.z & ((1 << LOG_CV) - 1);
  inout_ptr += cv_id * BLN + cipher_id * LN;

  const int groupID = threadIdx.x / GROUP_SIZE; // 0 ~ 7
  const int laneID = threadIdx.x % GROUP_SIZE; // 0 ~ 7
  const int C_N = 1 << LOG_N;
  const uint32_t batch_idx = blockIdx.y;
  base_inv += batch_idx * C_N;
  base_inv_ += batch_idx * C_N;

  const uint64_t* W = base_inv;
  const uint64_t* W_ = base_inv_;
  const uint64_t prime = primes[batch_idx];
  const uint64_t two_p = prime << 1;

  auto inout_matrix =
      reinterpret_cast<uint64_t(*)[8][GROUP_SIZE][C_N / (8 * GROUP_SIZE)]>(inout_ptr);

  const int N_init = NUM_GROUPS * blockIdx.x + laneID;

  uint64_t local[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    local[j] = inout_matrix[batch_idx][groupID][j][N_init];
  }

  LOCAL_INTT_RADIX<8>(
      local[0],
      local[1],
      local[2],
      local[3],
      local[4],
      local[5],
      local[6],
      local[7],
      8 + groupID,
      W,
      W_,
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

  LOCAL_INTT_RADIX<8>(
      local[0],
      local[1],
      local[2],
      local[3],
      local[4],
      local[5],
      local[6],
      local[7],
      1,
      W,
      W_,
      prime,
      two_p);
  for (int j = 0; j < 8; ++j) {
    if (local[j] >= prime) {
      local[j] -= prime;
    }
  }

#pragma unroll
  for (int j = 0; j < 8; ++j) {
    inout_matrix[batch_idx][j][groupID][N_init] = local[j];
  }
}

template <size_t LOG_N, size_t NUM_GROUPS, bool MODUP_SCALE>
__global__ void INTT64PointPhase2Scaled(
    uint64_t __restrict__* inout_ptr,
    const size_t LOG_CV,
    const size_t LN,
    const size_t BLN,
    const size_t curr_limbs,
    const size_t alpha,
    const size_t scalar_stride,
    const uint64_t __restrict__* base_inv,
    const uint64_t __restrict__* base_inv_,
    const uint64_t __restrict__* primes,
    const uint64_t __restrict__* scalars,
    const uint64_t __restrict__* scalar_shoups) {
  static_assert(NUM_GROUPS == 8);
  const int GROUP_SIZE = 8;

  auto cipher_id = blockIdx.z >> LOG_CV;
  auto cv_id = blockIdx.z & ((1 << LOG_CV) - 1);
  inout_ptr += cv_id * BLN + cipher_id * LN;

  const int groupID = threadIdx.x / GROUP_SIZE; // 0 ~ 7
  const int laneID = threadIdx.x % GROUP_SIZE; // 0 ~ 7
  const int C_N = 1 << LOG_N;
  const uint32_t batch_idx = blockIdx.y;
  base_inv += batch_idx * C_N;
  base_inv_ += batch_idx * C_N;

  const uint64_t* W = base_inv;
  const uint64_t* W_ = base_inv_;
  const uint64_t prime = primes[batch_idx];
  const uint64_t two_p = prime << 1;

  uint64_t scalar;
  uint64_t scalar_shoup;
  if constexpr (MODUP_SCALE) {
    const size_t begin_idx = (batch_idx / alpha) * alpha;
    const size_t group_size =
        (begin_idx + alpha > curr_limbs) ? (curr_limbs - begin_idx) : alpha;
    const size_t local_idx = batch_idx - begin_idx;
    const size_t scalar_row = begin_idx + group_size - 1;
    scalar = scalars[scalar_row * scalar_stride + local_idx];
    scalar_shoup = scalar_shoups[scalar_row * scalar_stride + local_idx];
  } else {
    scalar = scalars[batch_idx];
    scalar_shoup = scalar_shoups[batch_idx];
  }

  auto inout_matrix =
      reinterpret_cast<uint64_t(*)[8][GROUP_SIZE][C_N / (8 * GROUP_SIZE)]>(inout_ptr);

  const int N_init = NUM_GROUPS * blockIdx.x + laneID;

  uint64_t local[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    local[j] = inout_matrix[batch_idx][groupID][j][N_init];
  }

  LOCAL_INTT_RADIX<8>(
      local[0],
      local[1],
      local[2],
      local[3],
      local[4],
      local[5],
      local[6],
      local[7],
      8 + groupID,
      W,
      W_,
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

  LOCAL_INTT_RADIX<8>(
      local[0],
      local[1],
      local[2],
      local[3],
      local[4],
      local[5],
      local[6],
      local[7],
      1,
      W,
      W_,
      prime,
      two_p);
  for (int j = 0; j < 8; ++j) {
    if (local[j] >= prime) {
      local[j] -= prime;
    }
    local[j] = mul_and_reduce_shoup(local[j], scalar, scalar_shoup, prime);
    if (local[j] >= prime) {
      local[j] -= prime;
    }
  }

#pragma unroll
  for (int j = 0; j < 8; ++j) {
    inout_matrix[batch_idx][j][groupID][N_init] = local[j];
  }
}

template <size_t LOG_N, size_t NUM_GROUPS>
__global__ void NTT64PointPhase1(
    uint64_t __restrict__* inout_ptr,
    const size_t LOG_CV,
    const size_t LN,
    const size_t BLN,
    const uint64_t __restrict__* base_inv,
    const uint64_t __restrict__* base_inv_,
    const uint64_t __restrict__* primes) {
  static_assert(NUM_GROUPS == 8);
  const int GROUP_SIZE = 8;

  auto cipher_id = blockIdx.z >> LOG_CV;
  auto cv_id = blockIdx.z & ((1 << LOG_CV) - 1);
  inout_ptr += cv_id * BLN + cipher_id * LN;

  const int groupID = threadIdx.x / GROUP_SIZE;
  const int laneID = threadIdx.x % GROUP_SIZE;
  const int C_N = 1 << LOG_N;
  const uint32_t batch_id = blockIdx.y;
  base_inv += batch_id * C_N;
  base_inv_ += batch_id * C_N;

  const uint64_t* W = base_inv;
  const uint64_t* W_ = base_inv_;
  const uint64_t prime = primes[batch_id];
  const uint64_t two_p = prime << 1;

  auto inout_matrix =
      reinterpret_cast<uint64_t(*)[8][GROUP_SIZE][C_N / (8 * GROUP_SIZE)]>(inout_ptr);

  const int N_init = NUM_GROUPS * blockIdx.x + laneID;

  uint64_t local[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    local[j] = inout_matrix[batch_id][j][groupID][N_init];
  }

  ntt::local_radix<8>(
      local[0],
      local[1],
      local[2],
      local[3],
      local[4],
      local[5],
      local[6],
      local[7],
      1,
      W,
      W_,
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

  ntt::local_radix<8>(
      local[0],
      local[1],
      local[2],
      local[3],
      local[4],
      local[5],
      local[6],
      local[7],
      8 + groupID,
      W,
      W_,
      prime,
      two_p);

#pragma unroll
  for (int j = 0; j < 8; ++j) {
    inout_matrix[batch_id][groupID][j][N_init] = local[j];
  }
}

template <size_t LOG_N, size_t NUM_GROUPS>
__global__ void NTT64PointPhase1ModupMasked(
    uint64_t __restrict__* inout_ptr,
    const size_t LOG_CV,
    const size_t LN,
    const size_t BLN,
    const uint64_t __restrict__* base_inv,
    const uint64_t __restrict__* base_inv_,
    const uint64_t __restrict__* primes,
    const size_t curr_limbs,
    const size_t L,
    const size_t begin_idx,
    const size_t group_size) {
  const uint32_t limb_idx = blockIdx.y;
  if (limb_idx >= begin_idx && limb_idx < begin_idx + group_size) {
    return;
  }

  static_assert(NUM_GROUPS == 8);
  const int GROUP_SIZE = 8;

  auto cipher_id = blockIdx.z >> LOG_CV;
  auto cv_id = blockIdx.z & ((1 << LOG_CV) - 1);
  inout_ptr += cv_id * BLN + cipher_id * LN;

  const uint32_t prime_idx =
      limb_idx < curr_limbs ? limb_idx : L + (limb_idx - curr_limbs);
  const int groupID = threadIdx.x / GROUP_SIZE;
  const int laneID = threadIdx.x % GROUP_SIZE;
  const int C_N = 1 << LOG_N;
  base_inv += prime_idx * C_N;
  base_inv_ += prime_idx * C_N;

  const uint64_t* W = base_inv;
  const uint64_t* W_ = base_inv_;
  const uint64_t prime = primes[prime_idx];
  const uint64_t two_p = prime << 1;

  auto inout_matrix =
      reinterpret_cast<uint64_t(*)[8][GROUP_SIZE][C_N / (8 * GROUP_SIZE)]>(inout_ptr);

  const int N_init = NUM_GROUPS * blockIdx.x + laneID;

  uint64_t local[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    local[j] = inout_matrix[limb_idx][j][groupID][N_init];
  }

  ntt::local_radix<8>(
      local[0],
      local[1],
      local[2],
      local[3],
      local[4],
      local[5],
      local[6],
      local[7],
      1,
      W,
      W_,
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

  ntt::local_radix<8>(
      local[0],
      local[1],
      local[2],
      local[3],
      local[4],
      local[5],
      local[6],
      local[7],
      8 + groupID,
      W,
      W_,
      prime,
      two_p);

#pragma unroll
  for (int j = 0; j < 8; ++j) {
    inout_matrix[limb_idx][groupID][j][N_init] = local[j];
  }
}

template <size_t LOG_N, size_t NUM_GROUPS>
__global__ void NTT64PointPhase1ModupAllMasked(
    uint64_t __restrict__* inout_ptr,
    const size_t LOG_CV,
    const size_t LN,
    const size_t BLN,
    const uint64_t __restrict__* base_inv,
    const uint64_t __restrict__* base_inv_,
    const uint64_t __restrict__* primes,
    const size_t curr_limbs,
    const size_t L,
    const size_t alpha,
    const size_t num_moduli_after_modup) {
  const uint32_t physical_limb_idx = blockIdx.y;
  const uint32_t group_idx = physical_limb_idx / num_moduli_after_modup;
  const uint32_t limb_idx = physical_limb_idx - group_idx * num_moduli_after_modup;
  const uint32_t begin_idx = group_idx * alpha;
  const uint32_t group_size =
      min(static_cast<uint32_t>(alpha), static_cast<uint32_t>(curr_limbs - begin_idx));
  if (limb_idx >= begin_idx && limb_idx < begin_idx + group_size) {
    return;
  }

  static_assert(NUM_GROUPS == 8);
  const int GROUP_SIZE = 8;

  auto cipher_id = blockIdx.z >> LOG_CV;
  auto cv_id = blockIdx.z & ((1 << LOG_CV) - 1);
  inout_ptr += cv_id * BLN + cipher_id * LN;

  const uint32_t prime_idx =
      limb_idx < curr_limbs ? limb_idx : L + (limb_idx - curr_limbs);
  const int groupID = threadIdx.x / GROUP_SIZE;
  const int laneID = threadIdx.x % GROUP_SIZE;
  const int C_N = 1 << LOG_N;
  base_inv += prime_idx * C_N;
  base_inv_ += prime_idx * C_N;

  const uint64_t* W = base_inv;
  const uint64_t* W_ = base_inv_;
  const uint64_t prime = primes[prime_idx];
  const uint64_t two_p = prime << 1;

  auto inout_matrix =
      reinterpret_cast<uint64_t(*)[8][GROUP_SIZE][C_N / (8 * GROUP_SIZE)]>(inout_ptr);

  const int N_init = NUM_GROUPS * blockIdx.x + laneID;

  uint64_t local[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    local[j] = inout_matrix[physical_limb_idx][j][groupID][N_init];
  }

  ntt::local_radix<8>(
      local[0],
      local[1],
      local[2],
      local[3],
      local[4],
      local[5],
      local[6],
      local[7],
      1,
      W,
      W_,
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

  ntt::local_radix<8>(
      local[0],
      local[1],
      local[2],
      local[3],
      local[4],
      local[5],
      local[6],
      local[7],
      8 + groupID,
      W,
      W_,
      prime,
      two_p);

#pragma unroll
  for (int j = 0; j < 8; ++j) {
    inout_matrix[physical_limb_idx][groupID][j][N_init] = local[j];
  }
}

template <size_t LOG_N, size_t NUM_GROUPS>
__global__ void ModupStepTwoNTT64PointPhase1All(
    uint64_t __restrict__* out_ptr,
    const uint64_t __restrict__* in_ptr,
    const size_t LOG_CV,
    const size_t LN,
    const size_t BLN,
    const size_t L_INN,
    const uint64_t __restrict__* base_inv,
    const uint64_t __restrict__* base_inv_,
    const uint64_t __restrict__* primes,
    const uint64_t __restrict__* barrett_ratios,
    const uint64_t __restrict__* barrett_ks,
    const uint64_t __restrict__* prod_q_i_mod_q_js,
    const size_t prod_beta_stride,
    const size_t curr_limbs,
    const size_t L,
    const size_t alpha,
    const size_t num_moduli_after_modup) {
  const uint32_t physical_limb_idx = blockIdx.y;
  const uint32_t group_idx = physical_limb_idx / num_moduli_after_modup;
  const uint32_t limb_idx = physical_limb_idx - group_idx * num_moduli_after_modup;
  const uint32_t begin_idx = group_idx * alpha;
  const uint32_t group_size =
      min(static_cast<uint32_t>(alpha), static_cast<uint32_t>(curr_limbs - begin_idx));
  if (limb_idx >= begin_idx && limb_idx < begin_idx + group_size) {
    return;
  }

  static_assert(NUM_GROUPS == 8);
  constexpr int GROUP_SIZE = 8;
  constexpr int C_N = 1 << LOG_N;
  constexpr int COEFF_STRIDE = C_N / (8 * GROUP_SIZE);

  auto cipher_id = blockIdx.z >> LOG_CV;
  auto cv_id = blockIdx.z & ((1 << LOG_CV) - 1);
  out_ptr += cv_id * BLN + cipher_id * LN;
  in_ptr += cipher_id * L_INN;

  const uint32_t prime_idx =
      limb_idx < curr_limbs ? limb_idx : L + (limb_idx - curr_limbs);
  const uint32_t out_iter =
      limb_idx - ((limb_idx >= begin_idx + group_size) ? group_size : 0);
  const int groupID = threadIdx.x / GROUP_SIZE;
  const int laneID = threadIdx.x % GROUP_SIZE;
  const int N_init = NUM_GROUPS * blockIdx.x + laneID;

  base_inv += prime_idx * C_N;
  base_inv_ += prime_idx * C_N;
  const uint64_t prime = primes[prime_idx];
  const uint64_t two_p = prime << 1;
  const uint64_t barret_ratio = barrett_ratios[prime_idx];
  const uint64_t barret_k = barrett_ks[prime_idx];

  extern __shared__ uint64_t hat_mod_end_shared[];
  if (threadIdx.x < group_size) {
    const uint64_t* group_prod =
        prod_q_i_mod_q_js + group_idx * prod_beta_stride;
    hat_mod_end_shared[threadIdx.x] =
        group_prod[threadIdx.x + out_iter * alpha];
  }
  __syncthreads();

  auto out_matrix =
      reinterpret_cast<uint64_t(*)[8][GROUP_SIZE][COEFF_STRIDE]>(out_ptr);

  uint64_t local[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int coeff = j * (GROUP_SIZE * COEFF_STRIDE) +
        groupID * COEFF_STRIDE + N_init;
    uint128_t accum{0};
    for (int i = 0; i < group_size; i++) {
      const uint64_t op1 = in_ptr[(begin_idx + i) * C_N + coeff];
      const uint64_t op2 = hat_mod_end_shared[i];
      uint128_t out = mult_64_64_128(op1, op2);
      inplace_add_128_128(out, accum);
    }
    local[j] = barrett_reduction_128_64(accum, prime, barret_ratio, barret_k);
  }

  ntt::local_radix<8>(
      local[0],
      local[1],
      local[2],
      local[3],
      local[4],
      local[5],
      local[6],
      local[7],
      1,
      base_inv,
      base_inv_,
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

  ntt::local_radix<8>(
      local[0],
      local[1],
      local[2],
      local[3],
      local[4],
      local[5],
      local[6],
      local[7],
      8 + groupID,
      base_inv,
      base_inv_,
      prime,
      two_p);

#pragma unroll
  for (int j = 0; j < 8; ++j) {
    out_matrix[physical_limb_idx][groupID][j][N_init] = local[j];
  }
}

template <size_t LOG_N>
__global__ void NTTXPointPhase2(
    uint64_t* __restrict__ inout_ptr,
    const size_t LOG_CV,
    const size_t LN,
    const size_t BLN,
    const uint64_t* __restrict__ base_inv,
    const uint64_t* __restrict__ base_inv_,
    const uint64_t* __restrict__ primes) {
  constexpr size_t LOG_RADIX = LOG_N - 6;
  static_assert(LOG_RADIX >= 8);
  const int R1_RADIX = 64;
  const int DATA_SIZE = 1u << LOG_RADIX;
  const int kBLOCK_SIZE = 1u << (LOG_RADIX - 1);
  const int NUM_WARP = kBLOCK_SIZE / WARP_SIZE;

  auto batch_id = blockIdx.z >> LOG_CV;
  auto cv_id = blockIdx.z & ((1 << LOG_CV) - 1);
  inout_ptr += cv_id * BLN + batch_id * LN;

  const uint32_t poly_idx = blockIdx.y;
  base_inv += poly_idx * (R1_RADIX * DATA_SIZE);
  base_inv_ += poly_idx * (R1_RADIX * DATA_SIZE);
  const uint64_t prime = primes[poly_idx];
  const uint64_t two_p = prime << 1;

  auto g_row = reinterpret_cast<uint64_t(*)[R1_RADIX][DATA_SIZE]>(inout_ptr);
  __shared__ uint64_t tile[2][NUM_WARP][WARP_SIZE + 1];

  uint32_t stage_off = R1_RADIX + blockIdx.x;

  uint64_t i1 = g_row[poly_idx][blockIdx.x][threadIdx.x];
  uint64_t i2 = g_row[poly_idx][blockIdx.x][threadIdx.x + kBLOCK_SIZE];

  tile[0][threadIdx.x % NUM_WARP][threadIdx.x / NUM_WARP] = i1;
  tile[1][threadIdx.x % NUM_WARP][threadIdx.x / NUM_WARP] = i2;
  __syncthreads();

  uint32_t laneID = threadIdx.x % WARP_SIZE;
  uint32_t groupID = threadIdx.x / WARP_SIZE;

  i1 = tile[0][groupID][laneID];
  i2 = tile[1][groupID][laneID];

  ntt::warp_butterfly<6>(
      i1,
      i2,
      stage_off,
      laneID,
      base_inv,
      base_inv_,
      prime,
      two_p);

  tile[0][groupID][laneID] = i1;
  tile[1][groupID][laneID] = i2;

  __syncthreads();

  const int SECOND_GROUP_SIZE = DATA_SIZE / WARP_SIZE / 4;

  laneID = threadIdx.x % SECOND_GROUP_SIZE;
  groupID = threadIdx.x / SECOND_GROUP_SIZE;

  const uint32_t half_group = groupID >> 1;
  if ((groupID & 1) == 0) {
    i1 = tile[0][laneID][half_group];
    i2 = tile[0][laneID + SECOND_GROUP_SIZE][half_group];
  } else {
    i1 = tile[1][laneID][half_group];
    i2 = tile[1][laneID + SECOND_GROUP_SIZE][half_group];
  }

  stage_off = stage_off * 2 + groupID;
  ntt::warp_butterfly<LOG_RADIX - 6>(
      i1,
      i2,
      stage_off,
      laneID,
      base_inv,
      base_inv_,
      prime,
      two_p);

  for (int k = 0; k < 3; k++) {
    if (i1 >= prime)
      i1 -= prime;
    if (i2 >= prime)
      i2 -= prime;
  }

  auto g_row_out =
      reinterpret_cast<ulonglong2(*)[R1_RADIX][DATA_SIZE / 2]>(inout_ptr);
  ulonglong2 i12 = {i1, i2};
  g_row_out[poly_idx][blockIdx.x][threadIdx.x] = i12;
}

template <size_t LOG_N>
__global__ void NTTXPointPhase2ModupMasked(
    uint64_t* __restrict__ inout_ptr,
    const size_t LOG_CV,
    const size_t LN,
    const size_t BLN,
    const uint64_t* __restrict__ base_inv,
    const uint64_t* __restrict__ base_inv_,
    const uint64_t* __restrict__ primes,
    const size_t curr_limbs,
    const size_t L,
    const size_t begin_idx,
    const size_t group_size) {
  const uint32_t poly_idx = blockIdx.y;
  if (poly_idx >= begin_idx && poly_idx < begin_idx + group_size) {
    return;
  }

  constexpr size_t LOG_RADIX = LOG_N - 6;
  static_assert(LOG_RADIX >= 8);
  const int R1_RADIX = 64;
  const int DATA_SIZE = 1u << LOG_RADIX;
  const int kBLOCK_SIZE = 1u << (LOG_RADIX - 1);
  const int NUM_WARP = kBLOCK_SIZE / WARP_SIZE;

  auto batch_id = blockIdx.z >> LOG_CV;
  auto cv_id = blockIdx.z & ((1 << LOG_CV) - 1);
  inout_ptr += cv_id * BLN + batch_id * LN;

  const uint32_t prime_idx =
      poly_idx < curr_limbs ? poly_idx : L + (poly_idx - curr_limbs);
  base_inv += prime_idx * (R1_RADIX * DATA_SIZE);
  base_inv_ += prime_idx * (R1_RADIX * DATA_SIZE);
  const uint64_t prime = primes[prime_idx];
  const uint64_t two_p = prime << 1;

  auto g_row = reinterpret_cast<uint64_t(*)[R1_RADIX][DATA_SIZE]>(inout_ptr);
  __shared__ uint64_t tile[2][NUM_WARP][WARP_SIZE + 1];

  uint32_t stage_off = R1_RADIX + blockIdx.x;

  uint64_t i1 = g_row[poly_idx][blockIdx.x][threadIdx.x];
  uint64_t i2 = g_row[poly_idx][blockIdx.x][threadIdx.x + kBLOCK_SIZE];

  tile[0][threadIdx.x % NUM_WARP][threadIdx.x / NUM_WARP] = i1;
  tile[1][threadIdx.x % NUM_WARP][threadIdx.x / NUM_WARP] = i2;
  __syncthreads();

  uint32_t laneID = threadIdx.x % WARP_SIZE;
  uint32_t groupID = threadIdx.x / WARP_SIZE;

  i1 = tile[0][groupID][laneID];
  i2 = tile[1][groupID][laneID];

  ntt::warp_butterfly<6>(
      i1,
      i2,
      stage_off,
      laneID,
      base_inv,
      base_inv_,
      prime,
      two_p);

  tile[0][groupID][laneID] = i1;
  tile[1][groupID][laneID] = i2;

  __syncthreads();

  const int SECOND_GROUP_SIZE = DATA_SIZE / WARP_SIZE / 4;

  laneID = threadIdx.x % SECOND_GROUP_SIZE;
  groupID = threadIdx.x / SECOND_GROUP_SIZE;

  const uint32_t half_group = groupID >> 1;
  if ((groupID & 1) == 0) {
    i1 = tile[0][laneID][half_group];
    i2 = tile[0][laneID + SECOND_GROUP_SIZE][half_group];
  } else {
    i1 = tile[1][laneID][half_group];
    i2 = tile[1][laneID + SECOND_GROUP_SIZE][half_group];
  }

  stage_off = stage_off * 2 + groupID;
  ntt::warp_butterfly<LOG_RADIX - 6>(
      i1,
      i2,
      stage_off,
      laneID,
      base_inv,
      base_inv_,
      prime,
      two_p);

  for (int k = 0; k < 3; k++) {
    if (i1 >= prime)
      i1 -= prime;
    if (i2 >= prime)
      i2 -= prime;
  }

  auto g_row_out =
      reinterpret_cast<ulonglong2(*)[R1_RADIX][DATA_SIZE / 2]>(inout_ptr);
  ulonglong2 i12 = {i1, i2};
  g_row_out[poly_idx][blockIdx.x][threadIdx.x] = i12;
}

template <size_t LOG_N>
__global__ void NTTXPointPhase2ModupAllMasked(
    uint64_t* __restrict__ inout_ptr,
    const size_t LOG_CV,
    const size_t LN,
    const size_t BLN,
    const uint64_t* __restrict__ base_inv,
    const uint64_t* __restrict__ base_inv_,
    const uint64_t* __restrict__ primes,
    const size_t curr_limbs,
    const size_t L,
    const size_t alpha,
    const size_t num_moduli_after_modup) {
  const uint32_t physical_limb_idx = blockIdx.y;
  const uint32_t group_idx = physical_limb_idx / num_moduli_after_modup;
  const uint32_t limb_idx = physical_limb_idx - group_idx * num_moduli_after_modup;
  const uint32_t begin_idx = group_idx * alpha;
  const uint32_t group_size =
      min(static_cast<uint32_t>(alpha), static_cast<uint32_t>(curr_limbs - begin_idx));
  if (limb_idx >= begin_idx && limb_idx < begin_idx + group_size) {
    return;
  }

  constexpr size_t LOG_RADIX = LOG_N - 6;
  static_assert(LOG_RADIX >= 8);
  const int R1_RADIX = 64;
  const int DATA_SIZE = 1u << LOG_RADIX;
  const int kBLOCK_SIZE = 1u << (LOG_RADIX - 1);
  const int NUM_WARP = kBLOCK_SIZE / WARP_SIZE;

  auto batch_id = blockIdx.z >> LOG_CV;
  auto cv_id = blockIdx.z & ((1 << LOG_CV) - 1);
  inout_ptr += cv_id * BLN + batch_id * LN;

  const uint32_t prime_idx =
      limb_idx < curr_limbs ? limb_idx : L + (limb_idx - curr_limbs);
  base_inv += prime_idx * (R1_RADIX * DATA_SIZE);
  base_inv_ += prime_idx * (R1_RADIX * DATA_SIZE);
  const uint64_t prime = primes[prime_idx];
  const uint64_t two_p = prime << 1;

  auto g_row = reinterpret_cast<uint64_t(*)[R1_RADIX][DATA_SIZE]>(inout_ptr);
  __shared__ uint64_t tile[2][NUM_WARP][WARP_SIZE + 1];

  uint32_t stage_off = R1_RADIX + blockIdx.x;

  uint64_t i1 = g_row[physical_limb_idx][blockIdx.x][threadIdx.x];
  uint64_t i2 =
      g_row[physical_limb_idx][blockIdx.x][threadIdx.x + kBLOCK_SIZE];

  tile[0][threadIdx.x % NUM_WARP][threadIdx.x / NUM_WARP] = i1;
  tile[1][threadIdx.x % NUM_WARP][threadIdx.x / NUM_WARP] = i2;
  __syncthreads();

  uint32_t laneID = threadIdx.x % WARP_SIZE;
  uint32_t groupID = threadIdx.x / WARP_SIZE;

  i1 = tile[0][groupID][laneID];
  i2 = tile[1][groupID][laneID];

  ntt::warp_butterfly<6>(
      i1,
      i2,
      stage_off,
      laneID,
      base_inv,
      base_inv_,
      prime,
      two_p);

  tile[0][groupID][laneID] = i1;
  tile[1][groupID][laneID] = i2;

  __syncthreads();

  const int SECOND_GROUP_SIZE = DATA_SIZE / WARP_SIZE / 4;

  laneID = threadIdx.x % SECOND_GROUP_SIZE;
  groupID = threadIdx.x / SECOND_GROUP_SIZE;

  const uint32_t half_group = groupID >> 1;
  if ((groupID & 1) == 0) {
    i1 = tile[0][laneID][half_group];
    i2 = tile[0][laneID + SECOND_GROUP_SIZE][half_group];
  } else {
    i1 = tile[1][laneID][half_group];
    i2 = tile[1][laneID + SECOND_GROUP_SIZE][half_group];
  }

  stage_off = stage_off * 2 + groupID;
  ntt::warp_butterfly<LOG_RADIX - 6>(
      i1,
      i2,
      stage_off,
      laneID,
      base_inv,
      base_inv_,
      prime,
      two_p);

  for (int k = 0; k < 3; k++) {
    if (i1 >= prime)
      i1 -= prime;
    if (i2 >= prime)
      i2 -= prime;
  }

  auto g_row_out =
      reinterpret_cast<ulonglong2(*)[R1_RADIX][DATA_SIZE / 2]>(inout_ptr);
  ulonglong2 i12 = {i1, i2};
  g_row_out[physical_limb_idx][blockIdx.x][threadIdx.x] = i12;
}

} // namespace fhe

namespace at::native {

namespace {

constexpr int kNttNumGroups = 8;

size_t log_cv_for(size_t num_cv) {
  TORCH_INTERNAL_ASSERT(
      num_cv == 1 || num_cv == 2,
      "NTT_impl only supports num_cv == 1 or num_cv == 2");
  return num_cv == 1 ? 0 : 1;
}

template <size_t LOG_N>
void launch_iNTT_impl(
    uint64_t* out_ptr,
    uint64_t* in_ptr,
    size_t num_batch,
    size_t log_cv,
    size_t L_OUTN,
    size_t BL_OUTN,
    size_t L_INN,
    size_t BL_INN,
    size_t num_cv,
    size_t num_cipher,
    const uint64_t* param_primes_ptr,
    const uint64_t* inverse_power_of_roots_div_two_ptr,
    const uint64_t* inverse_scaled_power_of_roots_div_two_ptr,
    cudaStream_t stream) {
  constexpr size_t N = size_t{1} << LOG_N;
  constexpr size_t RADIX = 64;

  fhe::INTTXPointPhase1<LOG_N>
      <<<dim3(RADIX, num_batch, num_cv * num_cipher),
         N / RADIX / 2,
         0,
         stream>>>(
          in_ptr,
          out_ptr,
          log_cv,
          L_OUTN,
          BL_OUTN,
          L_INN,
          BL_INN,
          inverse_power_of_roots_div_two_ptr,
          inverse_scaled_power_of_roots_div_two_ptr,
          param_primes_ptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  fhe::INTT64PointPhase2<LOG_N, kNttNumGroups>
      <<<dim3(N / (kNttNumGroups * 8) / 8, num_batch, num_cv * num_cipher),
         kNttNumGroups * 8,
         0,
         stream>>>(
          out_ptr,
          log_cv,
          L_OUTN,
          BL_OUTN,
          inverse_power_of_roots_div_two_ptr,
          inverse_scaled_power_of_roots_div_two_ptr,
	          param_primes_ptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <size_t LOG_N, bool MODUP_SCALE>
void launch_iNTT_scaled_impl(
    uint64_t* out_ptr,
    uint64_t* in_ptr,
    size_t num_batch,
    size_t curr_limbs,
    size_t alpha,
    size_t scalar_stride,
    size_t log_cv,
    size_t L_OUTN,
    size_t BL_OUTN,
    size_t L_INN,
    size_t BL_INN,
    size_t num_cv,
    size_t num_cipher,
    const uint64_t* param_primes_ptr,
    const uint64_t* inverse_power_of_roots_div_two_ptr,
    const uint64_t* inverse_scaled_power_of_roots_div_two_ptr,
    const uint64_t* scalars,
    const uint64_t* scalar_shoups,
    cudaStream_t stream) {
  constexpr size_t N = size_t{1} << LOG_N;
  constexpr size_t RADIX = 64;

  fhe::INTTXPointPhase1<LOG_N>
      <<<dim3(RADIX, num_batch, num_cv * num_cipher),
         N / RADIX / 2,
         0,
         stream>>>(
          in_ptr,
          out_ptr,
          log_cv,
          L_OUTN,
          BL_OUTN,
          L_INN,
          BL_INN,
          inverse_power_of_roots_div_two_ptr,
          inverse_scaled_power_of_roots_div_two_ptr,
          param_primes_ptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  fhe::INTT64PointPhase2Scaled<LOG_N, kNttNumGroups, MODUP_SCALE>
      <<<dim3(N / (kNttNumGroups * 8) / 8, num_batch, num_cv * num_cipher),
         kNttNumGroups * 8,
         0,
         stream>>>(
          out_ptr,
          log_cv,
          L_OUTN,
          BL_OUTN,
          curr_limbs,
          alpha,
          scalar_stride,
          inverse_power_of_roots_div_two_ptr,
          inverse_scaled_power_of_roots_div_two_ptr,
          param_primes_ptr,
          scalars,
          scalar_shoups);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <size_t LOG_N>
void launch_NTT_impl(
    uint64_t* inout_ptr,
    size_t num_batch,
    size_t log_cv,
    size_t LN,
    size_t BLN,
    size_t num_cv,
    size_t num_cipher,
    const uint64_t* param_primes_ptr,
    const uint64_t* param_power_of_roots_shoup_ptr,
    const uint64_t* param_power_of_roots_ptr,
    cudaStream_t stream) {
  constexpr size_t N = size_t{1} << LOG_N;
  constexpr size_t RADIX = 64;

  fhe::NTT64PointPhase1<LOG_N, kNttNumGroups>
      <<<dim3(N / (kNttNumGroups * 8) / 8, num_batch, num_cv * num_cipher),
         kNttNumGroups * 8,
         0,
         stream>>>(
          inout_ptr,
          log_cv,
          LN,
          BLN,
          param_power_of_roots_ptr,
          param_power_of_roots_shoup_ptr,
          param_primes_ptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  fhe::NTTXPointPhase2<LOG_N>
      <<<dim3(RADIX, num_batch, num_cv * num_cipher),
         N / RADIX / 2,
         0,
         stream>>>(
          inout_ptr,
          log_cv,
          LN,
          BLN,
          param_power_of_roots_ptr,
          param_power_of_roots_shoup_ptr,
          param_primes_ptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <size_t LOG_N>
void launch_NTT_modup_masked_impl(
    uint64_t* inout_ptr,
    size_t num_batch,
    size_t curr_limbs,
    size_t L,
    size_t begin_idx,
    size_t group_size,
    size_t log_cv,
    size_t LN,
    size_t BLN,
    size_t num_cv,
    size_t num_cipher,
    const uint64_t* param_primes_ptr,
    const uint64_t* param_power_of_roots_shoup_ptr,
    const uint64_t* param_power_of_roots_ptr,
    cudaStream_t stream) {
  constexpr size_t N = size_t{1} << LOG_N;
  constexpr size_t RADIX = 64;

  fhe::NTT64PointPhase1ModupMasked<LOG_N, kNttNumGroups>
      <<<dim3(N / (kNttNumGroups * 8) / 8, num_batch, num_cv * num_cipher),
         kNttNumGroups * 8,
         0,
         stream>>>(
          inout_ptr,
          log_cv,
          LN,
          BLN,
          param_power_of_roots_ptr,
          param_power_of_roots_shoup_ptr,
          param_primes_ptr,
          curr_limbs,
          L,
          begin_idx,
          group_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  fhe::NTTXPointPhase2ModupMasked<LOG_N>
      <<<dim3(RADIX, num_batch, num_cv * num_cipher),
         N / RADIX / 2,
         0,
         stream>>>(
          inout_ptr,
          log_cv,
          LN,
          BLN,
          param_power_of_roots_ptr,
          param_power_of_roots_shoup_ptr,
          param_primes_ptr,
          curr_limbs,
          L,
          begin_idx,
          group_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <size_t LOG_N>
void launch_NTT_modup_all_masked_impl(
    uint64_t* inout_ptr,
    size_t beta,
    size_t curr_limbs,
    size_t L,
    size_t alpha,
    size_t num_moduli_after_modup,
    size_t log_cv,
    size_t LN,
    size_t BLN,
    size_t num_cv,
    size_t num_cipher,
    const uint64_t* param_primes_ptr,
    const uint64_t* param_power_of_roots_shoup_ptr,
    const uint64_t* param_power_of_roots_ptr,
    cudaStream_t stream) {
  constexpr size_t N = size_t{1} << LOG_N;
  constexpr size_t RADIX = 64;
  const size_t num_physical_limbs = beta * num_moduli_after_modup;

  fhe::NTT64PointPhase1ModupAllMasked<LOG_N, kNttNumGroups>
      <<<dim3(N / (kNttNumGroups * 8) / 8, num_physical_limbs, num_cv * num_cipher),
         kNttNumGroups * 8,
         0,
         stream>>>(
          inout_ptr,
          log_cv,
          LN,
          BLN,
          param_power_of_roots_ptr,
          param_power_of_roots_shoup_ptr,
          param_primes_ptr,
          curr_limbs,
          L,
          alpha,
          num_moduli_after_modup);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  fhe::NTTXPointPhase2ModupAllMasked<LOG_N>
      <<<dim3(RADIX, num_physical_limbs, num_cv * num_cipher),
         N / RADIX / 2,
         0,
         stream>>>(
          inout_ptr,
          log_cv,
          LN,
          BLN,
          param_power_of_roots_ptr,
          param_power_of_roots_shoup_ptr,
          param_primes_ptr,
          curr_limbs,
          L,
          alpha,
          num_moduli_after_modup);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <size_t LOG_N>
void launch_modup_step_two_ntt_all_impl(
    uint64_t* out_ptr,
    const uint64_t* in_ptr,
    size_t beta,
    size_t curr_limbs,
    size_t L,
    size_t alpha,
    size_t num_moduli_after_modup,
    size_t log_cv,
    size_t LN,
    size_t BLN,
    size_t L_INN,
    size_t num_cv,
    size_t num_cipher,
    const uint64_t* param_primes_ptr,
    const uint64_t* barrett_ratios,
    const uint64_t* barrett_ks,
    const uint64_t* prod_q_i_mod_q_js,
    size_t prod_beta_stride,
    const uint64_t* param_power_of_roots_shoup_ptr,
    const uint64_t* param_power_of_roots_ptr,
    cudaStream_t stream) {
  constexpr size_t N = size_t{1} << LOG_N;
  constexpr size_t RADIX = 64;
  const size_t num_physical_limbs = beta * num_moduli_after_modup;

  fhe::ModupStepTwoNTT64PointPhase1All<LOG_N, kNttNumGroups>
      <<<dim3(N / (kNttNumGroups * 8) / 8, num_physical_limbs, num_cv * num_cipher),
         kNttNumGroups * 8,
         alpha * sizeof(uint64_t),
         stream>>>(
          out_ptr,
          in_ptr,
          log_cv,
          LN,
          BLN,
          L_INN,
          param_power_of_roots_ptr,
          param_power_of_roots_shoup_ptr,
          param_primes_ptr,
          barrett_ratios,
          barrett_ks,
          prod_q_i_mod_q_js,
          prod_beta_stride,
          curr_limbs,
          L,
          alpha,
          num_moduli_after_modup);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  fhe::NTTXPointPhase2ModupAllMasked<LOG_N>
      <<<dim3(RADIX, num_physical_limbs, num_cv * num_cipher),
         N / RADIX / 2,
         0,
         stream>>>(
          out_ptr,
          log_cv,
          LN,
          BLN,
          param_power_of_roots_ptr,
          param_power_of_roots_shoup_ptr,
          param_primes_ptr,
          curr_limbs,
          L,
          alpha,
          num_moduli_after_modup);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace

void iNTT_impl(
    uint64_t* out_ptr,
    uint64_t* in_ptr,
    size_t num_batch,
    size_t N,
    size_t L_OUT,
    size_t L_IN,
    size_t num_cv,
    size_t num_cipher,
    const uint64_t* param_primes_ptr,
    const uint64_t* inverse_power_of_roots_div_two_ptr,
    const uint64_t* inverse_scaled_power_of_roots_div_two_ptr) {
  auto stream = at::cuda::getCurrentCUDAStream();
  auto LOG_CV = log_cv_for(num_cv);
  auto L_OUTN = L_OUT * N;
  auto BL_OUTN = L_OUTN * num_cipher;
  auto L_INN = L_IN * N;
  auto BL_INN = L_INN * num_cipher;

  if (N == (size_t{1} << 17)) {
    launch_iNTT_impl<17>(
        out_ptr,
        in_ptr,
        num_batch,
        LOG_CV,
        L_OUTN,
        BL_OUTN,
        L_INN,
        BL_INN,
        num_cv,
        num_cipher,
        param_primes_ptr,
        inverse_power_of_roots_div_two_ptr,
        inverse_scaled_power_of_roots_div_two_ptr,
        stream);
  } else if (N == (size_t{1} << 16)) {
    launch_iNTT_impl<16>(
        out_ptr,
        in_ptr,
        num_batch,
        LOG_CV,
        L_OUTN,
        BL_OUTN,
        L_INN,
        BL_INN,
        num_cv,
        num_cipher,
        param_primes_ptr,
        inverse_power_of_roots_div_two_ptr,
        inverse_scaled_power_of_roots_div_two_ptr,
        stream);
  } else if (N == (size_t{1} << 15)) {
    launch_iNTT_impl<15>(
        out_ptr,
        in_ptr,
        num_batch,
        LOG_CV,
        L_OUTN,
        BL_OUTN,
        L_INN,
        BL_INN,
        num_cv,
        num_cipher,
        param_primes_ptr,
        inverse_power_of_roots_div_two_ptr,
        inverse_scaled_power_of_roots_div_two_ptr,
        stream);
  } else if (N == (size_t{1} << 14)) {
    launch_iNTT_impl<14>(
        out_ptr,
        in_ptr,
        num_batch,
        LOG_CV,
        L_OUTN,
        BL_OUTN,
        L_INN,
        BL_INN,
        num_cv,
        num_cipher,
        param_primes_ptr,
        inverse_power_of_roots_div_two_ptr,
        inverse_scaled_power_of_roots_div_two_ptr,
        stream);
  } else {
    TORCH_INTERNAL_ASSERT(false, "Unsupported iNTT size");
  }
}

template <bool MODUP_SCALE>
static void iNTT_scaled_dispatch(
    uint64_t* out_ptr,
    uint64_t* in_ptr,
    size_t num_batch,
    size_t curr_limbs,
    size_t N,
    size_t L_OUT,
    size_t L_IN,
    size_t num_cv,
    size_t num_cipher,
    size_t alpha,
    size_t scalar_stride,
    const uint64_t* param_primes_ptr,
    const uint64_t* inverse_power_of_roots_div_two_ptr,
    const uint64_t* inverse_scaled_power_of_roots_div_two_ptr,
    const uint64_t* scalars,
    const uint64_t* scalar_shoups) {
  auto stream = at::cuda::getCurrentCUDAStream();
  auto LOG_CV = log_cv_for(num_cv);
  auto L_OUTN = L_OUT * N;
  auto BL_OUTN = L_OUTN * num_cipher;
  auto L_INN = L_IN * N;
  auto BL_INN = L_INN * num_cipher;

  if (N == (size_t{1} << 17)) {
    launch_iNTT_scaled_impl<17, MODUP_SCALE>(
        out_ptr,
        in_ptr,
        num_batch,
        curr_limbs,
        alpha,
        scalar_stride,
        LOG_CV,
        L_OUTN,
        BL_OUTN,
        L_INN,
        BL_INN,
        num_cv,
        num_cipher,
        param_primes_ptr,
        inverse_power_of_roots_div_two_ptr,
        inverse_scaled_power_of_roots_div_two_ptr,
        scalars,
        scalar_shoups,
        stream);
  } else if (N == (size_t{1} << 16)) {
    launch_iNTT_scaled_impl<16, MODUP_SCALE>(
        out_ptr,
        in_ptr,
        num_batch,
        curr_limbs,
        alpha,
        scalar_stride,
        LOG_CV,
        L_OUTN,
        BL_OUTN,
        L_INN,
        BL_INN,
        num_cv,
        num_cipher,
        param_primes_ptr,
        inverse_power_of_roots_div_two_ptr,
        inverse_scaled_power_of_roots_div_two_ptr,
        scalars,
        scalar_shoups,
        stream);
  } else if (N == (size_t{1} << 15)) {
    launch_iNTT_scaled_impl<15, MODUP_SCALE>(
        out_ptr,
        in_ptr,
        num_batch,
        curr_limbs,
        alpha,
        scalar_stride,
        LOG_CV,
        L_OUTN,
        BL_OUTN,
        L_INN,
        BL_INN,
        num_cv,
        num_cipher,
        param_primes_ptr,
        inverse_power_of_roots_div_two_ptr,
        inverse_scaled_power_of_roots_div_two_ptr,
        scalars,
        scalar_shoups,
        stream);
  } else if (N == (size_t{1} << 14)) {
    launch_iNTT_scaled_impl<14, MODUP_SCALE>(
        out_ptr,
        in_ptr,
        num_batch,
        curr_limbs,
        alpha,
        scalar_stride,
        LOG_CV,
        L_OUTN,
        BL_OUTN,
        L_INN,
        BL_INN,
        num_cv,
        num_cipher,
        param_primes_ptr,
        inverse_power_of_roots_div_two_ptr,
        inverse_scaled_power_of_roots_div_two_ptr,
        scalars,
        scalar_shoups,
        stream);
  } else {
    TORCH_INTERNAL_ASSERT(false, "Unsupported scaled iNTT size");
  }
}

void iNTT_scaled_impl(
    uint64_t* out_ptr,
    uint64_t* in_ptr,
    size_t num_batch,
    size_t N,
    size_t L_OUT,
    size_t L_IN,
    size_t num_cv,
    size_t num_cipher,
    const uint64_t* param_primes_ptr,
    const uint64_t* inverse_power_of_roots_div_two_ptr,
    const uint64_t* inverse_scaled_power_of_roots_div_two_ptr,
    const uint64_t* scalars,
    const uint64_t* scalar_shoups) {
  iNTT_scaled_dispatch<false>(
      out_ptr,
      in_ptr,
      num_batch,
      num_batch,
      N,
      L_OUT,
      L_IN,
      num_cv,
      num_cipher,
      1,
      num_batch,
      param_primes_ptr,
      inverse_power_of_roots_div_two_ptr,
      inverse_scaled_power_of_roots_div_two_ptr,
      scalars,
      scalar_shoups);
}

void iNTT_modup_scaled_impl(
    uint64_t* out_ptr,
    uint64_t* in_ptr,
    size_t curr_limbs,
    size_t N,
    size_t L_OUT,
    size_t L_IN,
    size_t num_cv,
    size_t num_cipher,
    size_t alpha,
    const uint64_t* param_primes_ptr,
    const uint64_t* inverse_power_of_roots_div_two_ptr,
    const uint64_t* inverse_scaled_power_of_roots_div_two_ptr,
    const uint64_t* scalars,
    const uint64_t* scalar_shoups,
    size_t scalar_stride) {
  iNTT_scaled_dispatch<true>(
      out_ptr,
      in_ptr,
      curr_limbs,
      curr_limbs,
      N,
      L_OUT,
      L_IN,
      num_cv,
      num_cipher,
      alpha,
      scalar_stride,
      param_primes_ptr,
      inverse_power_of_roots_div_two_ptr,
      inverse_scaled_power_of_roots_div_two_ptr,
      scalars,
      scalar_shoups);
}

void NTT_impl(
    uint64_t* inout_ptr,
    size_t num_batch,
    size_t N,
    size_t L,
    size_t num_cv,
    size_t num_cipher,
    const uint64_t* param_primes_ptr,
    const uint64_t* param_power_of_roots_shoup_ptr,
    const uint64_t* param_power_of_roots_ptr) {
  auto stream = at::cuda::getCurrentCUDAStream();

  auto LOG_CV = log_cv_for(num_cv);
  auto LN = L*N;
  auto BLN = LN * num_cipher;
  if (N == (size_t{1} << 17)) {
    launch_NTT_impl<17>(
        inout_ptr,
        num_batch,
        LOG_CV,
        LN,
        BLN,
        num_cv,
        num_cipher,
        param_primes_ptr,
        param_power_of_roots_shoup_ptr,
        param_power_of_roots_ptr,
        stream);
  } else if (N == (size_t{1} << 16)) {
    launch_NTT_impl<16>(
        inout_ptr,
        num_batch,
        LOG_CV,
        LN,
        BLN,
        num_cv,
        num_cipher,
        param_primes_ptr,
        param_power_of_roots_shoup_ptr,
        param_power_of_roots_ptr,
        stream);
  } else if (N == (size_t{1} << 15)) {
    launch_NTT_impl<15>(
        inout_ptr,
        num_batch,
        LOG_CV,
        LN,
        BLN,
        num_cv,
        num_cipher,
        param_primes_ptr,
        param_power_of_roots_shoup_ptr,
        param_power_of_roots_ptr,
        stream);
  } else if (N == (size_t{1} << 14)) {
    launch_NTT_impl<14>(
        inout_ptr,
        num_batch,
        LOG_CV,
        LN,
        BLN,
        num_cv,
        num_cipher,
        param_primes_ptr,
        param_power_of_roots_shoup_ptr,
        param_power_of_roots_ptr,
        stream);
  } else {
    TORCH_INTERNAL_ASSERT(false, "Unsupported NTT size");
  }
}

void NTT_modup_masked_impl(
    uint64_t* inout_ptr,
    size_t num_batch,
    size_t curr_limbs,
    size_t N,
    size_t L,
    size_t begin_idx,
    size_t group_size,
    size_t L_OUT,
    size_t num_cv,
    size_t num_cipher,
    const uint64_t* param_primes_ptr,
    const uint64_t* param_power_of_roots_shoup_ptr,
    const uint64_t* param_power_of_roots_ptr) {
  auto stream = at::cuda::getCurrentCUDAStream();

  auto LOG_CV = log_cv_for(num_cv);
  auto LN = L_OUT * N;
  auto BLN = LN * num_cipher;
  if (N == (size_t{1} << 17)) {
    launch_NTT_modup_masked_impl<17>(
        inout_ptr,
        num_batch,
        curr_limbs,
        L,
        begin_idx,
        group_size,
        LOG_CV,
        LN,
        BLN,
        num_cv,
        num_cipher,
        param_primes_ptr,
        param_power_of_roots_shoup_ptr,
        param_power_of_roots_ptr,
        stream);
  } else if (N == (size_t{1} << 16)) {
    launch_NTT_modup_masked_impl<16>(
        inout_ptr,
        num_batch,
        curr_limbs,
        L,
        begin_idx,
        group_size,
        LOG_CV,
        LN,
        BLN,
        num_cv,
        num_cipher,
        param_primes_ptr,
        param_power_of_roots_shoup_ptr,
        param_power_of_roots_ptr,
        stream);
  } else if (N == (size_t{1} << 15)) {
    launch_NTT_modup_masked_impl<15>(
        inout_ptr,
        num_batch,
        curr_limbs,
        L,
        begin_idx,
        group_size,
        LOG_CV,
        LN,
        BLN,
        num_cv,
        num_cipher,
        param_primes_ptr,
        param_power_of_roots_shoup_ptr,
        param_power_of_roots_ptr,
        stream);
  } else if (N == (size_t{1} << 14)) {
    launch_NTT_modup_masked_impl<14>(
        inout_ptr,
        num_batch,
        curr_limbs,
        L,
        begin_idx,
        group_size,
        LOG_CV,
        LN,
        BLN,
        num_cv,
        num_cipher,
        param_primes_ptr,
        param_power_of_roots_shoup_ptr,
        param_power_of_roots_ptr,
        stream);
  } else {
    TORCH_INTERNAL_ASSERT(false, "Unsupported NTT size");
  }
}

void NTT_modup_all_masked_impl(
    uint64_t* inout_ptr,
    size_t beta,
    size_t curr_limbs,
    size_t N,
    size_t L,
    size_t alpha,
    size_t num_moduli_after_modup,
    size_t L_OUT,
    size_t num_cv,
    size_t num_cipher,
    const uint64_t* param_primes_ptr,
    const uint64_t* param_power_of_roots_shoup_ptr,
    const uint64_t* param_power_of_roots_ptr) {
  auto stream = at::cuda::getCurrentCUDAStream();

  auto LOG_CV = log_cv_for(num_cv);
  auto LN = L_OUT * N;
  auto BLN = LN * num_cipher;
  if (N == (size_t{1} << 17)) {
    launch_NTT_modup_all_masked_impl<17>(
        inout_ptr,
        beta,
        curr_limbs,
        L,
        alpha,
        num_moduli_after_modup,
        LOG_CV,
        LN,
        BLN,
        num_cv,
        num_cipher,
        param_primes_ptr,
        param_power_of_roots_shoup_ptr,
        param_power_of_roots_ptr,
        stream);
  } else if (N == (size_t{1} << 16)) {
    launch_NTT_modup_all_masked_impl<16>(
        inout_ptr,
        beta,
        curr_limbs,
        L,
        alpha,
        num_moduli_after_modup,
        LOG_CV,
        LN,
        BLN,
        num_cv,
        num_cipher,
        param_primes_ptr,
        param_power_of_roots_shoup_ptr,
        param_power_of_roots_ptr,
        stream);
  } else if (N == (size_t{1} << 15)) {
    launch_NTT_modup_all_masked_impl<15>(
        inout_ptr,
        beta,
        curr_limbs,
        L,
        alpha,
        num_moduli_after_modup,
        LOG_CV,
        LN,
        BLN,
        num_cv,
        num_cipher,
        param_primes_ptr,
        param_power_of_roots_shoup_ptr,
        param_power_of_roots_ptr,
        stream);
  } else if (N == (size_t{1} << 14)) {
    launch_NTT_modup_all_masked_impl<14>(
        inout_ptr,
        beta,
        curr_limbs,
        L,
        alpha,
        num_moduli_after_modup,
        LOG_CV,
        LN,
        BLN,
        num_cv,
        num_cipher,
        param_primes_ptr,
        param_power_of_roots_shoup_ptr,
        param_power_of_roots_ptr,
        stream);
  } else {
    TORCH_INTERNAL_ASSERT(false, "Unsupported NTT size");
  }
}

void modup_step_two_ntt_all_impl(
    uint64_t* out_ptr,
    const uint64_t* in_ptr,
    size_t beta,
    size_t curr_limbs,
    size_t N,
    size_t L,
    size_t alpha,
    size_t num_moduli_after_modup,
    size_t L_OUT,
    size_t L_IN,
    size_t num_cv,
    size_t num_cipher,
    const uint64_t* param_primes_ptr,
    const uint64_t* barrett_ratios,
    const uint64_t* barrett_ks,
    const uint64_t* prod_q_i_mod_q_js,
    size_t prod_beta_stride,
    const uint64_t* param_power_of_roots_shoup_ptr,
    const uint64_t* param_power_of_roots_ptr) {
  auto stream = at::cuda::getCurrentCUDAStream();
  auto LOG_CV = log_cv_for(num_cv);
  auto LN = L_OUT * N;
  auto BLN = LN * num_cipher;
  auto L_INN = L_IN * N;

  if (N == (size_t{1} << 17)) {
    launch_modup_step_two_ntt_all_impl<17>(
        out_ptr,
        in_ptr,
        beta,
        curr_limbs,
        L,
        alpha,
        num_moduli_after_modup,
        LOG_CV,
        LN,
        BLN,
        L_INN,
        num_cv,
        num_cipher,
        param_primes_ptr,
        barrett_ratios,
        barrett_ks,
        prod_q_i_mod_q_js,
        prod_beta_stride,
        param_power_of_roots_shoup_ptr,
        param_power_of_roots_ptr,
        stream);
  } else if (N == (size_t{1} << 16)) {
    launch_modup_step_two_ntt_all_impl<16>(
        out_ptr,
        in_ptr,
        beta,
        curr_limbs,
        L,
        alpha,
        num_moduli_after_modup,
        LOG_CV,
        LN,
        BLN,
        L_INN,
        num_cv,
        num_cipher,
        param_primes_ptr,
        barrett_ratios,
        barrett_ks,
        prod_q_i_mod_q_js,
        prod_beta_stride,
        param_power_of_roots_shoup_ptr,
        param_power_of_roots_ptr,
        stream);
  } else if (N == (size_t{1} << 15)) {
    launch_modup_step_two_ntt_all_impl<15>(
        out_ptr,
        in_ptr,
        beta,
        curr_limbs,
        L,
        alpha,
        num_moduli_after_modup,
        LOG_CV,
        LN,
        BLN,
        L_INN,
        num_cv,
        num_cipher,
        param_primes_ptr,
        barrett_ratios,
        barrett_ks,
        prod_q_i_mod_q_js,
        prod_beta_stride,
        param_power_of_roots_shoup_ptr,
        param_power_of_roots_ptr,
        stream);
  } else if (N == (size_t{1} << 14)) {
    launch_modup_step_two_ntt_all_impl<14>(
        out_ptr,
        in_ptr,
        beta,
        curr_limbs,
        L,
        alpha,
        num_moduli_after_modup,
        LOG_CV,
        LN,
        BLN,
        L_INN,
        num_cv,
        num_cipher,
        param_primes_ptr,
        barrett_ratios,
        barrett_ks,
        prod_q_i_mod_q_js,
        prod_beta_stride,
        param_power_of_roots_shoup_ptr,
        param_power_of_roots_ptr,
        stream);
  } else {
    TORCH_INTERNAL_ASSERT(false, "Unsupported modup NTT size");
  }
}

} // namespace at::native
