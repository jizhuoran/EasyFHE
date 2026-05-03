#pragma once

#include "ATen/native/fhe/cuda/Utils.cuh"

namespace fhe {
__device__ void butt_intt_local(
    uint64_t& x,
    uint64_t& y,
    const uint64_t& w,
    const uint64_t& w_,
    const uint64_t& p) {
  const uint64_t two_p = 2 * p;
  const uint64_t T = two_p - y + x;
  uint64_t new_x = x + y;
  if (new_x >= two_p)
    new_x -= two_p;
  if (T & 1)
    new_x += p;
  x = (new_x >> 1);
  y = mul_and_reduce_shoup(T, w, w_, p);
}

__device__ void intt_warp_butterfly(
    uint64_t& i1,
    uint64_t& i2,
    uint64_t& stage_off,
    const size_t num_rounds,
    const size_t localID,
    const uint64_t* base_inv,
    const uint64_t* base_inv_,
    const uint64_t prime) {
  butt_intt_local(
      i1,
      i2,
      base_inv[stage_off + localID],
      base_inv_[stage_off + localID],
      prime);

#pragma unroll
  for (int shift = 1; shift < num_rounds;
       ++shift) { // offsets: 2, 4, 8, 16, ...
    const unsigned offset = 1u << shift; // 2^shift
    const bool lower_half = (localID & (offset - 1)) < (offset >> 1);

    // choose the value to exchange, then shuffle across the offset‑distance
    auto tmp = lower_half ? i2 : i1;
    tmp = __shfl_xor_sync(0xFFFFFFFF, tmp, offset >> 1);

    lower_half ? i2 = tmp : i1 = tmp; // exchange values

    // advance table pointer for the current NTT stage
    stage_off >>= 1; // equivalent to stage_off /= 2
    const unsigned idx = stage_off + (localID >> shift);

    butt_intt_local(i1, i2, base_inv[idx], base_inv_[idx], prime);
  }
}

template <size_t LOG_RADIX, size_t LOG_N>
__global__ void INTTXPointPhase1(
    uint64_t* in_ptr,
    uint64_t* out_ptr,
    const size_t LOG_CV,
    const size_t L_OUTN,
    const size_t BL_OUTN,
    const size_t L_INN,
    const size_t BL_INN,
    const uint64_t* base_inv,
    const uint64_t* base_inv_,
    const uint64_t* primes) {
  const int R2_RADIX = 64;
  const int DATA_SIZE = 1u << LOG_RADIX; // 1024 for 2^10, 2048 for 2^11
  const int kBLOCK_SIZE = 1u << (LOG_RADIX - 1);

  auto cipher_id = blockIdx.z >> LOG_CV;
  auto cv_id = blockIdx.z & ((1 << LOG_CV) - 1);
  in_ptr += (cv_id * BL_INN + cipher_id * L_INN);
  out_ptr += (cv_id * BL_OUTN + cipher_id * L_OUTN);


  auto batch_idx = blockIdx.y;
  base_inv += batch_idx * (1u << LOG_N);
  base_inv_ += batch_idx * (1u << LOG_N);

  auto in_row =
      reinterpret_cast<ulonglong2(*)[R2_RADIX][DATA_SIZE / 2]>(in_ptr);
  ulonglong2 i12 = in_row[batch_idx][blockIdx.x][threadIdx.x];
  uint64_t i1 = i12.x;
  uint64_t i2 = i12.y;

  uint64_t stage_off = (1 << (LOG_N - 1)) + blockIdx.x * blockDim.x;
  intt_warp_butterfly(
      i1,
      i2,
      stage_off,
      6,
      threadIdx.x,
      base_inv,
      base_inv_,
      primes[batch_idx]);

  auto localID = threadIdx.x % WARP_SIZE;
  auto groupID = threadIdx.x / WARP_SIZE;
  __shared__ uint64_t tile[kBLOCK_SIZE / WARP_SIZE][2 * WARP_SIZE + 1];
  tile[groupID][localID] = i1;
  tile[groupID][localID + WARP_SIZE] = i2;
  __syncthreads();
  auto tile2 = reinterpret_cast<uint64_t(*)[4 * WARP_SIZE + 2]>(tile);
  auto SECOND_GROUP_SIZE = DATA_SIZE / WARP_SIZE / 4; // 8 for 2^10, 16 for 2^11

  localID = threadIdx.x / SECOND_GROUP_SIZE; // 0 ~ 63
  groupID = threadIdx.x % SECOND_GROUP_SIZE; // 0 ~ 7 for 2^10, 0 ~ 15 for 2^11

  i1 = tile2[groupID][localID];
  i2 = tile2[groupID][localID + WARP_SIZE * 2 + 1];

  stage_off = stage_off >> 1; // equivalent to stage_off /= 2
  intt_warp_butterfly(
      i1,
      i2,
      stage_off,
      LOG_RADIX - 6, // num_rounds
      groupID, // laneID
      base_inv,
      base_inv_,
      primes[batch_idx]);

  auto tile3 = reinterpret_cast<uint64_t(*)[2 * WARP_SIZE + 1]>(tile);
  tile3[groupID][localID] = i1;
  tile3[groupID + SECOND_GROUP_SIZE][localID] = i2;
  __syncthreads();

  auto out_row = reinterpret_cast<uint64_t(*)[R2_RADIX][DATA_SIZE]>(out_ptr);
  out_row[batch_idx][blockIdx.x][threadIdx.x] =
      tile3[threadIdx.x / (2 * WARP_SIZE)][threadIdx.x % (2 * WARP_SIZE)];
  out_row[batch_idx][blockIdx.x][threadIdx.x + kBLOCK_SIZE] =
      tile3[(threadIdx.x + kBLOCK_SIZE) / (2 * WARP_SIZE)]
           [threadIdx.x % (2 * WARP_SIZE)];
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
    const size_t tw_off,
    const uint64_t* W,
    const uint64_t* W_,
    const uint64_t prime) {
  if constexpr (radix >= 8) {
    butt_intt_local(local0, local1, W[4 * tw_off], W_[4 * tw_off], prime);
    butt_intt_local(
        local2, local3, W[4 * tw_off + 1], W_[4 * tw_off + 1], prime);
    butt_intt_local(
        local4, local5, W[4 * tw_off + 2], W_[4 * tw_off + 2], prime);
    butt_intt_local(
        local6, local7, W[4 * tw_off + 3], W_[4 * tw_off + 3], prime);
  }

  if constexpr (radix >= 4) {
    butt_intt_local(local0, local2, W[2 * tw_off], W_[2 * tw_off], prime);
    butt_intt_local(local1, local3, W[2 * tw_off], W_[2 * tw_off], prime);
    butt_intt_local(
        local4, local6, W[2 * tw_off + 1], W_[2 * tw_off + 1], prime);
    butt_intt_local(
        local5, local7, W[2 * tw_off + 1], W_[2 * tw_off + 1], prime);
  }
  if constexpr (radix >= 2) {
    butt_intt_local(local0, local4, W[tw_off], W_[tw_off], prime);
    butt_intt_local(local1, local5, W[tw_off], W_[tw_off], prime);
    butt_intt_local(local2, local6, W[tw_off], W_[tw_off], prime);
    butt_intt_local(local3, local7, W[tw_off], W_[tw_off], prime);
  }
}

template <size_t LOG_N, size_t NUM_GROUPS>
__global__ void INTT64PointPhase2(
    uint64_t __restrict__* inout_ptr,
    const int N,
    const size_t LOG_CV,
    const size_t LN,
    const size_t BLN,
    const uint64_t __restrict__* base_inv,
    const uint64_t __restrict__* base_inv_,
    const uint64_t __restrict__* primes) {
  const int GROUP_SIZE = 8;
  // const int BLOCK_SIZE = NUM_GROUPS * GROUP_SIZE;

    auto cipher_id = blockIdx.z >> LOG_CV;
  auto cv_id = blockIdx.z & ((1 << LOG_CV) - 1);
  inout_ptr += (cv_id * BLN + cipher_id * LN);


  // Thread and group configuration
  // threadIdx.x in 0 ~ 63, blockIdx.x in 0 ~ 127
  const int groupID = threadIdx.x / GROUP_SIZE; // 0 ~ 7
  const int laneID = threadIdx.x % GROUP_SIZE; // 0 ~ 7
  const int C_N = 1 << LOG_N;
  // Prime-related parameters
  auto batch_idx = blockIdx.y;
  base_inv += batch_idx * C_N;
  base_inv_ += batch_idx * C_N;

  const uint64_t* W = base_inv;
  const uint64_t* W_ = base_inv_;
  const uint64_t prime = primes[batch_idx];

  // Reshape input for 4D access
  auto inout_matrix =
      reinterpret_cast<uint64_t(*)[8][GROUP_SIZE][C_N / (8 * GROUP_SIZE)]>(inout_ptr);

  // Compute logical index for each thread
  const int N_init = NUM_GROUPS * blockIdx.x + laneID;

  // 1. Load 8 elements per thread from global to local
  uint64_t local[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    local[j] = inout_matrix[batch_idx][groupID][j][N_init];
  }

  // 2. First local NTT (radix-8)
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
      prime);

  __shared__ uint64_t transpose_matrix[NUM_GROUPS][GROUP_SIZE + 1][8 + 1];

// 3. Transpose: store local results to shared memory for all-to-all exchange
// auto transpose_matrix = reinterpret_cast<uint64_t
// (*)[GROUP_SIZE+1][8+1]>(temp);
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    transpose_matrix[laneID][j][groupID] = local[j];
  }
  __syncthreads();

// 4. Reload data after transpose
#pragma unroll
  for (int l = 0; l < 8; ++l) {
    local[l] = transpose_matrix[laneID][groupID][l];
  }

  // 5. Second local NTT (radix-8)
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
      prime);
  for (int j = 0; j < 8; ++j) {
    if (local[j] >= prime) {
      local[j] -= prime;
    }
  }

// 6. Store results back to global memory
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    inout_matrix[batch_idx][j][groupID][N_init] = local[j];
  }
}

__device__ __inline__ void butt_ntt_local(
    uint64_t& a,
    uint64_t& b,
    const uint64_t& w,
    const uint64_t& w_,
    const uint64_t p) {
  uint64_t two_p = 2 * p;
  uint64_t U = mul_and_reduce_shoup(b, w, w_, p);
  if (a >= two_p)
    a -= two_p;
  b = a + (two_p - U);
  a += U;
}

template <uint8_t radix>
__device__ __forceinline__ void LOCAL_NTT_RADIX(
    uint64_t& local0,
    uint64_t& local1,
    uint64_t& local2,
    uint64_t& local3,
    uint64_t& local4,
    uint64_t& local5,
    uint64_t& local6,
    uint64_t& local7,
    const size_t tw_off,
    const uint64_t* W,
    const uint64_t* W_,
    const uint64_t prime) {
  if constexpr (radix >= 8) {
    butt_ntt_local(local0, local4, W[tw_off], W_[tw_off], prime);
    butt_ntt_local(local1, local5, W[tw_off], W_[tw_off], prime);
    butt_ntt_local(local2, local6, W[tw_off], W_[tw_off], prime);
    butt_ntt_local(local3, local7, W[tw_off], W_[tw_off], prime);
  }

  if constexpr (radix >= 4) {
    butt_ntt_local(local0, local2, W[2 * tw_off], W_[2 * tw_off], prime);
    butt_ntt_local(local1, local3, W[2 * tw_off], W_[2 * tw_off], prime);
    butt_ntt_local(
        local4, local6, W[2 * tw_off + 1], W_[2 * tw_off + 1], prime);
    butt_ntt_local(
        local5, local7, W[2 * tw_off + 1], W_[2 * tw_off + 1], prime);
  }
  if constexpr (radix >= 2) {
    butt_ntt_local(local0, local1, W[4 * tw_off], W_[4 * tw_off], prime);
    butt_ntt_local(
        local2, local3, W[4 * tw_off + 1], W_[4 * tw_off + 1], prime);
    butt_ntt_local(
        local4, local5, W[4 * tw_off + 2], W_[4 * tw_off + 2], prime);
    butt_ntt_local(
        local6, local7, W[4 * tw_off + 3], W_[4 * tw_off + 3], prime);
  }
}

template <size_t LOG_N, size_t NUM_GROUPS>
__global__ void NTT64PointPhase1(
    uint64_t __restrict__* inout_ptr,
    const size_t N,
    const size_t LOG_CV,
    const size_t LN,
    const size_t BLN,
    const uint64_t __restrict__* base_inv,
    const uint64_t __restrict__* base_inv_,
    const uint64_t __restrict__* primes) {
  const int GROUP_SIZE = 8;
  // const int BLOCK_SIZE = NUM_GROUPS * GROUP_SIZE;

  auto cipher_id = blockIdx.z >> LOG_CV;
  auto cv_id = blockIdx.z & ((1 << LOG_CV) - 1);
  inout_ptr += (cv_id * BLN + cipher_id * LN);

  // Thread and group configuration
  const int groupID = threadIdx.x / GROUP_SIZE;
  const int laneID = threadIdx.x % GROUP_SIZE;
  const int C_N = 1 << LOG_N;
  // Prime-related parameters
  auto batch_id = blockIdx.y;
  base_inv += batch_id * C_N;
  base_inv_ += batch_id * C_N;

  const uint64_t* W = base_inv;
  const uint64_t* W_ = base_inv_;
  const uint64_t prime = primes[batch_id];

  // Reshape input for 4D access
  auto inout_matrix =
      reinterpret_cast<uint64_t(*)[8][GROUP_SIZE][C_N / (8 * GROUP_SIZE)]>(inout_ptr);

  // Compute logical index for each thread
  const int N_init = NUM_GROUPS * blockIdx.x + laneID;

  // 1. Load 8 elements per thread from global to local
  uint64_t local[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    local[j] = inout_matrix[batch_id][j][groupID][N_init];
  }

  // 2. First local NTT (radix-8)
  LOCAL_NTT_RADIX<8>(
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
      prime);

  __shared__ uint64_t transpose_matrix[NUM_GROUPS][GROUP_SIZE + 1][8 + 1];

// 3. Transpose: store local results to shared memory for all-to-all exchange
// auto transpose_matrix = reinterpret_cast<uint64_t
// (*)[GROUP_SIZE+1][8+1]>(temp);
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    transpose_matrix[laneID][j][groupID] = local[j];
  }
  __syncthreads();

// 4. Reload data after transpose
#pragma unroll
  for (int l = 0; l < 8; ++l) {
    local[l] = transpose_matrix[laneID][groupID][l];
  }

  // 5. Second local NTT (radix-8)
  LOCAL_NTT_RADIX<8>(
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
      prime);

// 6. Store results back to global memory
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    inout_matrix[batch_id][groupID][j][N_init] = local[j];
  }
}

__device__ void warp_butterfly(
    uint64_t& i1,
    uint64_t& i2,
    uint64_t& stage_off,
    const size_t num_rounds,
    const size_t laneID,
    const uint64_t* base_inv,
    const uint64_t* base_inv_,
    const uint64_t prime) {
  butt_ntt_local(i1, i2, base_inv[stage_off], base_inv_[stage_off], prime);

#pragma unroll
  for (int shift = num_rounds - 2; shift >= 0;
       --shift) { // offsets: 16, 8, 4, 2, 1
    const unsigned offset = 1u << shift; // 2^shift
    const bool lower_half =
        (laneID & offset) == 0; // bit‑test replaces costly “%”

    // choose the value to exchange, then shuffle across the offset‑distance
    auto tmp = lower_half ? i2 : i1;
    tmp = __shfl_xor_sync(0xFFFFFFFF, tmp, offset);

    // commit the exchanged value
    if (lower_half)
      i2 = tmp;
    else
      i1 = tmp;

    // advance table pointer for the current NTT stage
    stage_off <<= 1; // equivalent to stage_off *= 2
    const unsigned idx =
        stage_off + (laneID >> shift); // laneID/offset ⇒ laneID >> shift

    butt_ntt_local(i1, i2, base_inv[idx], base_inv_[idx], prime);
  }
}

template <size_t LOG_RADIX>
__global__ void NTTXPointPhase2(
    uint64_t* inout_ptr,
    const size_t LOG_CV,
    const size_t LN,
    const size_t BLN,
    const uint64_t* base_inv,
    const uint64_t* base_inv_,
    const uint64_t* primes) {
  const int R1_RADIX = 64;
  const int DATA_SIZE = 1u << LOG_RADIX;
  const int kBLOCK_SIZE = 1u << (LOG_RADIX - 1);
  const int NUM_WARP = kBLOCK_SIZE / WARP_SIZE;

  auto batch_id = blockIdx.z >> LOG_CV;
  auto cv_id = blockIdx.z & ((1 << LOG_CV) - 1);
  inout_ptr += cv_id * BLN + batch_id * LN;

  auto poly_idx = blockIdx.y;
  base_inv += poly_idx * (R1_RADIX * DATA_SIZE);
  base_inv_ += poly_idx * (R1_RADIX * DATA_SIZE);

  auto g_row = reinterpret_cast<uint64_t(*)[R1_RADIX][DATA_SIZE]>(inout_ptr);
  __shared__ uint64_t
      tile[kBLOCK_SIZE / WARP_SIZE][2 * WARP_SIZE + 1]; //[16][64]

  uint64_t stage_off = R1_RADIX + blockIdx.x;

  uint64_t i1 = g_row[poly_idx][blockIdx.x][threadIdx.x];
  uint64_t i2 = g_row[poly_idx][blockIdx.x][threadIdx.x + kBLOCK_SIZE];

  tile[threadIdx.x % NUM_WARP][(threadIdx.x / NUM_WARP)] = i1;
  tile[threadIdx.x % NUM_WARP][(threadIdx.x / NUM_WARP) + WARP_SIZE] = i2;
  __syncthreads();

  auto laneID = threadIdx.x % WARP_SIZE;
  auto groupID = threadIdx.x / WARP_SIZE;

  i1 = tile[groupID][laneID];
  i2 = tile[groupID][laneID + WARP_SIZE];

  warp_butterfly(
      i1,
      i2,
      stage_off,
      6, // num_rounds
      laneID,
      base_inv,
      base_inv_,
      primes[poly_idx]);

  tile[groupID][laneID * 2] = i1;
  tile[groupID][laneID * 2 + 1] = i2;

  __syncthreads();

  const int SECOND_GROUP_SIZE = DATA_SIZE / WARP_SIZE / 4;

  laneID = threadIdx.x % SECOND_GROUP_SIZE;
  groupID = threadIdx.x / SECOND_GROUP_SIZE;

  i1 = tile[laneID][groupID];
  i2 = tile[laneID + SECOND_GROUP_SIZE][groupID];

  stage_off = stage_off * 2 + groupID;
  warp_butterfly(
      i1,
      i2,
      stage_off,
      LOG_RADIX - 6, // num_rounds
      laneID,
      base_inv,
      base_inv_,
      primes[poly_idx]);

  for (int k = 0; k < 3; k++) {
    if (i1 >= primes[poly_idx])
      i1 -= primes[poly_idx];
    if (i2 >= primes[poly_idx])
      i2 -= primes[poly_idx];
  }

  auto g_row_out = reinterpret_cast<ulonglong2(*)[R1_RADIX][DATA_SIZE/2]>(inout_ptr);
  ulonglong2 i12 = {i1, i2};
  g_row_out[poly_idx][blockIdx.x][threadIdx.x] = i12;
  // g_row[poly_idx][blockIdx.x][threadIdx.x * 2] = i1;
  // g_row[poly_idx][blockIdx.x][threadIdx.x * 2 + 1] = i2;

  return;
}

} // namespace fhe

namespace at::native {

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
  const int R1_RADIX = 64;
  auto stream = at::cuda::getCurrentCUDAStream();
  const int NUM_GROUPS = 8;
  auto LOG_CV = (num_cv == 1) ? 0 : 1; // 1 for 2, 0 for 1
  auto L_OUTN = L_OUT * N;
  auto BL_OUTN = L_OUTN * num_cipher;
  auto L_INN = L_IN * N;
  auto BL_INN = L_INN * num_cipher;

  if (N == 1 << 17) {
    fhe::INTTXPointPhase1<17 - 6, 17>
        <<<dim3(R1_RADIX, num_batch, num_cv * num_cipher), N / R1_RADIX / 2>>>(
            in_ptr,
            out_ptr,
            LOG_CV,
            L_OUTN,
            BL_OUTN,
            L_INN,
            BL_INN,
            inverse_power_of_roots_div_two_ptr,
            inverse_scaled_power_of_roots_div_two_ptr,
            param_primes_ptr);

    fhe::INTT64PointPhase2<17, NUM_GROUPS>
        <<<dim3(N / (NUM_GROUPS * 8) / 8, num_batch, num_cv * num_cipher),
           NUM_GROUPS * 8>>>(
            out_ptr,
            N,
            LOG_CV,
            L_OUTN,
            BL_OUTN,
            inverse_power_of_roots_div_two_ptr,
            inverse_scaled_power_of_roots_div_two_ptr,
            param_primes_ptr);
  } else if (N == 1 << 16) {
    fhe::INTTXPointPhase1<16 - 6, 16>
        <<<dim3(R1_RADIX, num_batch, num_cv * num_cipher), N / R1_RADIX / 2>>>(
            in_ptr,
            out_ptr,
            LOG_CV,
            L_OUTN,
            BL_OUTN,
            L_INN,
            BL_INN,
            inverse_power_of_roots_div_two_ptr,
            inverse_scaled_power_of_roots_div_two_ptr,
            param_primes_ptr);

    fhe::INTT64PointPhase2<16, NUM_GROUPS>
        <<<dim3(N / (NUM_GROUPS * 8) / 8, num_batch, num_cv * num_cipher),
           NUM_GROUPS * 8>>>(
            out_ptr,
            N,
            LOG_CV,
            L_OUTN,
            BL_OUTN,
            inverse_power_of_roots_div_two_ptr,
            inverse_scaled_power_of_roots_div_two_ptr,
            param_primes_ptr);
  } else if (N == 1 << 15) {
    fhe::INTTXPointPhase1<15 - 6, 15>
        <<<dim3(R1_RADIX, num_batch, num_cv * num_cipher), N / R1_RADIX / 2>>>(
            in_ptr,
            out_ptr,
            LOG_CV,
            L_OUTN,
            BL_OUTN,
            L_INN,
            BL_INN,
            inverse_power_of_roots_div_two_ptr,
            inverse_scaled_power_of_roots_div_two_ptr,
            param_primes_ptr);

    fhe::INTT64PointPhase2<15, NUM_GROUPS>
        <<<dim3(N / (NUM_GROUPS * 8) / 8, num_batch, num_cv * num_cipher),
           NUM_GROUPS * 8>>>(
            out_ptr,
            N,
            LOG_CV,
            L_OUTN,
            BL_OUTN,
            inverse_power_of_roots_div_two_ptr,
            inverse_scaled_power_of_roots_div_two_ptr,
            param_primes_ptr);
  } else if (N == 1 << 14) {
    fhe::INTTXPointPhase1<14 - 6, 14>
        <<<dim3(R1_RADIX, num_batch, num_cv * num_cipher), N / R1_RADIX / 2>>>(
            in_ptr,
            out_ptr,
            LOG_CV,
            L_OUTN,
            BL_OUTN,
            L_INN,
            BL_INN,
            inverse_power_of_roots_div_two_ptr,
            inverse_scaled_power_of_roots_div_two_ptr,
            param_primes_ptr);

    fhe::INTT64PointPhase2<14, NUM_GROUPS>
        <<<dim3(N / (NUM_GROUPS * 8) / 8, num_batch, num_cv * num_cipher),
           NUM_GROUPS * 8>>>(
            out_ptr,
            N,
            LOG_CV,
            L_OUTN,
            BL_OUTN,
            inverse_power_of_roots_div_two_ptr,
            inverse_scaled_power_of_roots_div_two_ptr,
            param_primes_ptr);
  } else {
    TORCH_INTERNAL_ASSERT(false, "Unsupported iNTT size");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
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
  const int NUM_GROUPS = 8;
  const int R1_RADIX = 64;

  auto LOG_CV = (num_cv == 1) ? 0 : 1; // 1 for 2, 0 for 1
  auto LN = L*N;
  auto BLN = LN * num_cipher;
  if (N == 1 << 17) {
    fhe::NTT64PointPhase1<17, NUM_GROUPS>
        <<<dim3(N / (NUM_GROUPS * 8) / 8, num_batch, num_cv * num_cipher), NUM_GROUPS * 8>>>(
            inout_ptr,
            N,
            LOG_CV,
            LN,
            BLN,
            param_power_of_roots_ptr,
            param_power_of_roots_shoup_ptr,
            param_primes_ptr);
    fhe::NTTXPointPhase2<11><<<dim3(R1_RADIX, num_batch, num_cv * num_cipher), N / R1_RADIX / 2>>>(
        inout_ptr,
            LOG_CV,
            LN,
            BLN,
        param_power_of_roots_ptr,
        param_power_of_roots_shoup_ptr,
        param_primes_ptr);
  } else if (N == 1 << 16) {
    fhe::NTT64PointPhase1<16, NUM_GROUPS>
        <<<dim3(N / (NUM_GROUPS * 8) / 8, num_batch, num_cv * num_cipher), NUM_GROUPS * 8>>>(
            inout_ptr,
            N,
            LOG_CV,
            LN,
            BLN,
            param_power_of_roots_ptr,
            param_power_of_roots_shoup_ptr,
            param_primes_ptr);
    fhe::NTTXPointPhase2<10><<<dim3(R1_RADIX, num_batch, num_cv * num_cipher), N / R1_RADIX / 2>>>(
        inout_ptr,
                    LOG_CV,
            LN,
            BLN,
        param_power_of_roots_ptr,
        param_power_of_roots_shoup_ptr,
        param_primes_ptr);
  } else if (N == 1 << 15) {
    fhe::NTT64PointPhase1<15, NUM_GROUPS>
        <<<dim3(N / (NUM_GROUPS * 8) / 8, num_batch, num_cv * num_cipher), NUM_GROUPS * 8>>>(
            inout_ptr,
            N,
            LOG_CV,
            LN,
            BLN,
            param_power_of_roots_ptr,
            param_power_of_roots_shoup_ptr,
            param_primes_ptr);
    fhe::NTTXPointPhase2<9><<<dim3(R1_RADIX, num_batch, num_cv * num_cipher), N / R1_RADIX / 2>>>(
        inout_ptr,
                    LOG_CV,
            LN,
            BLN,
        param_power_of_roots_ptr,
        param_power_of_roots_shoup_ptr,
        param_primes_ptr);
  } else if (N == 1 << 14) {
    fhe::NTT64PointPhase1<14, NUM_GROUPS>
        <<<dim3(N / (NUM_GROUPS * 8) / 8, num_batch, num_cv * num_cipher), NUM_GROUPS * 8>>>(
            inout_ptr,
            N,
            LOG_CV,
            LN,
            BLN,
            param_power_of_roots_ptr,
            param_power_of_roots_shoup_ptr,
            param_primes_ptr);
    fhe::NTTXPointPhase2<8><<<dim3(R1_RADIX, num_batch, num_cv * num_cipher), N / R1_RADIX / 2>>>(
        inout_ptr,
                    LOG_CV,
            LN,
            BLN,
        param_power_of_roots_ptr,
        param_power_of_roots_shoup_ptr,
        param_primes_ptr);
  } else {
    TORCH_INTERNAL_ASSERT(false, "Unsupported NTT size");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace at::native

// void iNTT_impl_old(
//   uint64_t* out_ptr,
//   uint64_t* in_ptr,
//   int64_t batch,
//   int64_t param_degree,
//   const uint64_t* param_primes_ptr,
//   const uint64_t* inverse_power_of_roots_div_two_ptr,
//   const uint64_t* inverse_scaled_power_of_roots_div_two_ptr) {
// dim3 gridDim(2048);
// dim3 blockDim(256);
// const int per_thread_ntt_size = 8;
// const int first_stage_radix_size = 256;
// const int second_radix_size = param_degree / first_stage_radix_size;
// const int pad = 4;
// const int per_thread_storage =
//     blockDim.x * per_thread_ntt_size * sizeof(uint64_t);
// auto stream = at::cuda::getCurrentCUDAStream();
// fhe::Intt8PointPerThreadPhase2OoP<<<
//     gridDim,
//     blockDim,
//     per_thread_storage,
//     stream>>>(
//     in_ptr,
//     first_stage_radix_size,
//     batch,
//     param_degree,
//     0,
//     // curr_limbs,
//     second_radix_size / per_thread_ntt_size,
//     inverse_power_of_roots_div_two_ptr,
//     inverse_scaled_power_of_roots_div_two_ptr,
//     param_primes_ptr,
//     out_ptr);
// fhe::Intt8PointPerThreadPhase1OoP<<<
//     gridDim,
//     (first_stage_radix_size / 8) * pad,
//     (first_stage_radix_size + pad + 1) * pad * sizeof(uint64_t),
//     stream>>>(
//     out_ptr,
//     1,
//     batch,
//     param_degree,
//     0,
//     // curr_limbs,
//     pad,
//     first_stage_radix_size / 8,
//     inverse_power_of_roots_div_two_ptr,
//     inverse_scaled_power_of_roots_div_two_ptr,
//     param_primes_ptr,
//     out_ptr);
// }

// void NTT_impl_old(
//   uint64_t* inout_ptr,
//   int64_t batch,
//   int64_t param_degree,
//   const uint64_t* param_primes_ptr,
//   const uint64_t* param_power_of_roots_shoup_ptr,
//   const uint64_t* param_power_of_roots_ptr) {
// dim3 gridDim(2048);
// dim3 blockDim(256);
// const int per_thread_ntt_size = 8;
// const int first_stage_radix_size = 256;
// const int second_radix_size = param_degree / first_stage_radix_size;
// const int pad = 4;
// const int per_thread_storage =
//     blockDim.x * per_thread_ntt_size * sizeof(uint64_t);

// auto stream = at::cuda::getCurrentCUDAStream();
// fhe::Ntt8PointPerThreadPhase1<<<
//     gridDim,
//     (first_stage_radix_size / 8) * pad,
//     (first_stage_radix_size + pad + 1) * pad * sizeof(uint64_t),
//     stream>>>(
//     inout_ptr,
//     inout_ptr,
//     1,
//     batch,
//     param_degree,
//     pad,
//     first_stage_radix_size / per_thread_ntt_size,
//     param_power_of_roots_ptr,
//     param_power_of_roots_shoup_ptr,
//     param_primes_ptr);
// fhe::Ntt8PointPerThreadPhase2<<<
//     gridDim,
//     blockDim.x,
//     per_thread_storage,
//     stream>>>(
//     inout_ptr,
//     inout_ptr,
//     first_stage_radix_size,
//     batch,
//     param_degree,
//     second_radix_size / per_thread_ntt_size,
//     param_power_of_roots_ptr,
//     param_power_of_roots_shoup_ptr,
//     param_primes_ptr);
// C10_CUDA_KERNEL_LAUNCH_CHECK();
// }

// void NTT_except_some_range_impl(
//     uint64_t* op_ptr,
//     int64_t batch,
//     int64_t N,
//     int64_t curr_limbs,
//     int64_t L,
//     int64_t start_prime_idx,
//     int64_t excluded_range_start,
//     int64_t excluded_range_size,
//     const Tensor& power_of_roots_shoup,
//     const Tensor& primes,
//     const Tensor& power_of_roots) {
//   auto excluded_range_end = excluded_range_start + excluded_range_size;
//   dim3 grid(2048);
//   dim3 block(256);
//   const int per_thread_ntt_size = 8;
//   const int first_stage_radix_size = 256;
//   const int second_radix_size = N / first_stage_radix_size;
//   const int pad = 4;
//   const int per_thread_storage =
//       block.x * per_thread_ntt_size * sizeof(uint64_t);

//   auto param_power_of_roots_shoup_ptr =
//       reinterpret_cast<uint64_t*>(power_of_roots_shoup.data_ptr<uint64_t>());
//   auto param_primes_ptr =
//       reinterpret_cast<uint64_t*>(primes.data_ptr<uint64_t>());
//   auto param_power_of_roots_ptr =
//       reinterpret_cast<uint64_t*>(power_of_roots.data_ptr<uint64_t>());
//   auto stream = at::cuda::getCurrentCUDAStream();
//   fhe::Ntt8PointPerThreadPhase1ExcludeSomeRange<<<
//       dim3(N / 8 / ((first_stage_radix_size / 8) * pad), batch),
//       (first_stage_radix_size / 8) * pad,
//       (first_stage_radix_size + pad + 1) * pad * sizeof(uint64_t),
//       stream>>>(
//       op_ptr,
//       1,
//       batch,
//       N,
//       start_prime_idx,
//       excluded_range_start,
//       excluded_range_end,
//       curr_limbs,
//       pad,
//       first_stage_radix_size / per_thread_ntt_size,
//       param_power_of_roots_ptr,
//       param_power_of_roots_shoup_ptr,
//       param_primes_ptr);
//   fhe::Ntt8PointPerThreadPhase2ExcludeSomeRange<<<
//       dim3(N / 8 / block.x, batch),
//       block.x,
//       per_thread_storage,
//       stream>>>(
//       op_ptr,
//       first_stage_radix_size,
//       batch,
//       N,
//       start_prime_idx,
//       excluded_range_start,
//       excluded_range_end,
//       curr_limbs,
//       second_radix_size / per_thread_ntt_size,
//       param_power_of_roots_ptr,
//       param_power_of_roots_shoup_ptr,
//       param_primes_ptr);
//   C10_CUDA_KERNEL_LAUNCH_CHECK();
// }

// __global__ void Intt8PointPerThreadPhase2OoP(
//   uint64_t* in,
//   const int m,
//   const int num_prime,
//   const int N,
//   const int start_prime_idx,
//   // const int ceil_curr_limbs,
//   const int radix,
//   const uint64_t* base_inv,
//   const uint64_t* base_inv_,
//   const uint64_t* primes,
//   uint64_t* out) {
// extern __shared__ uint64_t temp[];
// int set = threadIdx.x / radix;
// for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (N / 8 * num_prime);
//      i += blockDim.x * gridDim.x) {
//   // size of a block
//   uint64_t local[8];
//   int t = N / 2 / m;
//   // prime idx
//   int np_idx = i / (N / 8);
//   int prime_idx = np_idx;
//   // index in N/2 range
//   int N_idx = i % (N / 8);
//   // i'th block
//   int m_idx = N_idx / (t / 4);
//   int t_idx = N_idx % (t / 4);
//   // base address
//   uint64_t* in_addr = in + np_idx * N;
//   uint64_t* out_addr = out + np_idx * N;
//   const uint64_t* prime_table = primes;
//   uint64_t prime = prime_table[prime_idx];
//   int N_init = 2 * m_idx * t + t_idx;
//   __syncthreads();
//   for (int j = 0; j < 8; j++) {
//     temp[set * 8 * radix + t_idx + t / 4 * j] =
//         *(in_addr + N_init + t / 4 * j);
//   }
//   __syncthreads();
//   for (int l = 0; l < 8; l++) {
//     local[l] = temp[set * 8 * radix + 8 * t_idx + l];
//   }
//   int tw_idx = m + m_idx;
//   int tw_idx2 = (t / 4) * tw_idx + t_idx;
//   const uint64_t* WInv = base_inv + N * prime_idx;
//   const uint64_t* WInv_ = base_inv_ + N * prime_idx;
//   for (int j = 0; j < 4; j++) {
//     butt_intt_local(
//         local[2 * j],
//         local[2 * j + 1],
//         WInv[4 * tw_idx2 + j],
//         WInv_[4 * tw_idx2 + j],
//         prime);
//   }
//   for (int j = 0; j < 2; j++) {
//     butt_intt_local(
//         local[4 * j],
//         local[4 * j + 2],
//         WInv[2 * tw_idx2 + j],
//         WInv_[2 * tw_idx2 + j],
//         prime);
//     butt_intt_local(
//         local[4 * j + 1],
//         local[4 * j + 3],
//         WInv[2 * tw_idx2 + j],
//         WInv_[2 * tw_idx2 + j],
//         prime);
//   }
//   for (int j = 0; j < 4; j++) {
//     butt_intt_local(
//         local[j], local[j + 4], WInv[tw_idx2], WInv_[tw_idx2], prime);
//   }
//   int tail = 0;
//   __syncthreads();
//   for (int l = 0; l < 8; l++) {
//     temp[set * 8 * radix + 8 * t_idx + l] = local[l];
//   }
//   __syncthreads();
// #pragma unroll
//   for (int j = t / 32, k = 32; j > 0; j >>= 3, k *= 8) {
//     int m_idx2 = t_idx / (k / 4);
//     int t_idx2 = t_idx % (k / 4);
//     for (int l = 0; l < 8; l++) {
//       local[l] =
//           temp[set * 8 * radix + 2 * m_idx2 * k + t_idx2 + (k / 4) * l];
//     }
//     tw_idx2 = j * tw_idx + m_idx2;
//     for (int l = 0; l < 4; l++) {
//       butt_intt_local(
//           local[2 * l],
//           local[2 * l + 1],
//           WInv[4 * tw_idx2 + l],
//           WInv_[4 * tw_idx2 + l],
//           prime);
//     }
//     for (int l = 0; l < 2; l++) {
//       butt_intt_local(
//           local[4 * l],
//           local[4 * l + 2],
//           WInv[2 * tw_idx2 + l],
//           WInv_[2 * tw_idx2 + l],
//           prime);
//       butt_intt_local(
//           local[4 * l + 1],
//           local[4 * l + 3],
//           WInv[2 * tw_idx2 + l],
//           WInv_[2 * tw_idx2 + l],
//           prime);
//     }
//     for (int l = 0; l < 4; l++) {
//       butt_intt_local(
//           local[l], local[l + 4], WInv[tw_idx2], WInv_[tw_idx2], prime);
//     }
//     for (int l = 0; l < 8; l++) {
//       temp[set * 8 * radix + 2 * m_idx2 * k + t_idx2 + (k / 4) * l] =
//           local[l];
//     }
//     if (j == 2)
//       tail = 1;
//     if (j == 4)
//       tail = 2;
//     __syncthreads();
//   }
//   if (tail == 1) {
//     for (int j = 0; j < 8; j++) {
//       local[j] = temp[set * 8 * radix + t_idx + t / 4 * j];
//     }
//     butt_intt_local(local[0], local[4], WInv[tw_idx], WInv_[tw_idx], prime);
//     butt_intt_local(local[1], local[5], WInv[tw_idx], WInv_[tw_idx], prime);
//     butt_intt_local(local[2], local[6], WInv[tw_idx], WInv_[tw_idx], prime);
//     butt_intt_local(local[3], local[7], WInv[tw_idx], WInv_[tw_idx], prime);
//   } else if (tail == 2) {
//     for (int j = 0; j < 8; j++) {
//       local[j] = temp[set * 8 * radix + t_idx + t / 4 * j];
//     }
//     butt_intt_local(
//         local[0], local[2], WInv[2 * tw_idx], WInv_[2 * tw_idx], prime);
//     butt_intt_local(
//         local[1], local[3], WInv[2 * tw_idx], WInv_[2 * tw_idx], prime);
//     butt_intt_local(
//         local[4],
//         local[6],
//         WInv[2 * tw_idx + 1],
//         WInv_[2 * tw_idx + 1],
//         prime);
//     butt_intt_local(
//         local[5],
//         local[7],
//         WInv[2 * tw_idx + 1],
//         WInv_[2 * tw_idx + 1],
//         prime);
//     butt_intt_local(local[0], local[4], WInv[tw_idx], WInv_[tw_idx], prime);
//     butt_intt_local(local[1], local[5], WInv[tw_idx], WInv_[tw_idx], prime);
//     butt_intt_local(local[2], local[6], WInv[tw_idx], WInv_[tw_idx], prime);
//     butt_intt_local(local[3], local[7], WInv[tw_idx], WInv_[tw_idx], prime);
//   }
//   for (int j = 0; j < 8; j++) {
//     *(out_addr + N_init + t / 4 * j) = local[j];
//   }
// }
// }

// __global__ void Intt8PointPerThreadPhase1OoP(
//   uint64_t* in,
//   const int m,
//   const int num_prime,
//   const int N,
//   const int start_prime_idx,
//   // const int ceil_curr_limbs,
//   int pad,
//   int radix,
//   const uint64_t* base_inv,
//   const uint64_t* base_inv_,
//   const uint64_t* primes,
//   uint64_t* out) {
// extern __shared__ uint64_t temp[];
// int Warp_t = threadIdx.x % pad;
// int WarpID = threadIdx.x / pad;
// for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (N / 8 * num_prime);
//      i += blockDim.x * gridDim.x) {
//   // size of a block
//   uint64_t local[8];
//   int t = N / 2 / m;
//   // prime idx
//   int np_idx = i / (N / 8) + start_prime_idx;
//   int prime_idx = np_idx ;
//   // index in N/2 range
//   int N_idx = i % (N / 8);
//   // i'th block
//   int m_idx = N_idx / (t / 4);
//   int t_idx = N_idx % (t / 4);
//   // base address
//   const uint64_t* in_addr = in + np_idx * N;
//   uint64_t* out_addr = out + np_idx * N;
//   const uint64_t* prime_table = primes;
//   const uint64_t* WInv = base_inv + N * prime_idx;
//   const uint64_t* WInv_ = base_inv_ + N * prime_idx;
//   uint64_t prime = prime_table[prime_idx];
//   int N_init =
//       2 * t / radix * WarpID + Warp_t + pad * (t_idx / (radix * pad));
//   for (int j = 0; j < 8; j++) {
//     local[j] = *(in_addr + N_init + t / 4 / radix * j);
//   }
//   int eradix = 8 * radix;
//   int tw_idx = m + m_idx;
//   int tw_idx2 = radix * tw_idx + WarpID;
//   for (int j = 0; j < 4; j++) {
//     butt_intt_local(
//         local[2 * j],
//         local[2 * j + 1],
//         WInv[4 * tw_idx2 + j],
//         WInv_[4 * tw_idx2 + j],
//         prime);
//   }
//   for (int j = 0; j < 2; j++) {
//     butt_intt_local(
//         local[4 * j],
//         local[4 * j + 2],
//         WInv[2 * tw_idx2 + j],
//         WInv_[2 * tw_idx2 + j],
//         prime);
//     butt_intt_local(
//         local[4 * j + 1],
//         local[4 * j + 3],
//         WInv[2 * tw_idx2 + j],
//         WInv_[2 * tw_idx2 + j],
//         prime);
//   }
//   for (int j = 0; j < 4; j++) {
//     butt_intt_local(
//         local[j], local[j + 4], WInv[tw_idx2], WInv_[tw_idx2], prime);
//   }
//   for (int j = 0; j < 8; j++) {
//     temp[Warp_t * (eradix + pad) + 8 * WarpID + j] = local[j];
//   }
//   int tail = 0;
//   __syncthreads();
// #pragma unroll
//   for (int j = radix / 8, k = 32; j > 0; j >>= 3, k *= 8) {
//     int m_idx2 = WarpID / (k / 4);
//     int t_idx2 = WarpID % (k / 4);
//     for (int l = 0; l < 8; l++) {
//       local[l] = temp
//           [(eradix + pad) * Warp_t + 2 * m_idx2 * k + t_idx2 + (k / 4) * l];
//     }
//     int tw_idx2 = j * tw_idx + m_idx2;
//     for (int l = 0; l < 4; l++) {
//       butt_intt_local(
//           local[2 * l],
//           local[2 * l + 1],
//           WInv[4 * tw_idx2 + l],
//           WInv_[4 * tw_idx2 + l],
//           prime);
//     }
//     for (int l = 0; l < 2; l++) {
//       butt_intt_local(
//           local[4 * l],
//           local[4 * l + 2],
//           WInv[2 * tw_idx2 + l],
//           WInv_[2 * tw_idx2 + l],
//           prime);
//       butt_intt_local(
//           local[4 * l + 1],
//           local[4 * l + 3],
//           WInv[2 * tw_idx2 + l],
//           WInv_[2 * tw_idx2 + l],
//           prime);
//     }
//     for (int l = 0; l < 4; l++) {
//       butt_intt_local(
//           local[l], local[l + 4], WInv[tw_idx2], WInv_[tw_idx2], prime);
//     }
//     for (int l = 0; l < 8; l++) {
//       temp[(eradix + pad) * Warp_t + 2 * m_idx2 * k + t_idx2 + (k / 4) * l] =
//           local[l];
//     }
//     if (j == 2)
//       tail = 1;
//     if (j == 4)
//       tail = 2;
//     __syncthreads();
//   }
//   if (radix < 8)
//     tail = (radix == 4) ? 2 : 1;
//   for (int l = 0; l < 8; l++) {
//     local[l] = temp[Warp_t * (eradix + pad) + WarpID + radix * l];
//   }
//   if (tail == 1) {
//     butt_intt_local(local[0], local[4], WInv[tw_idx], WInv_[tw_idx], prime);
//     butt_intt_local(local[1], local[5], WInv[tw_idx], WInv_[tw_idx], prime);
//     butt_intt_local(local[2], local[6], WInv[tw_idx], WInv_[tw_idx], prime);
//     butt_intt_local(local[3], local[7], WInv[tw_idx], WInv_[tw_idx], prime);
//   } else if (tail == 2) {
//     butt_intt_local(
//         local[0], local[2], WInv[2 * tw_idx], WInv_[2 * tw_idx], prime);
//     butt_intt_local(
//         local[1], local[3], WInv[2 * tw_idx], WInv_[2 * tw_idx], prime);
//     butt_intt_local(
//         local[4],
//         local[6],
//         WInv[2 * tw_idx + 1],
//         WInv_[2 * tw_idx + 1],
//         prime);
//     butt_intt_local(
//         local[5],
//         local[7],
//         WInv[2 * tw_idx + 1],
//         WInv_[2 * tw_idx + 1],
//         prime);
//     butt_intt_local(local[0], local[4], WInv[tw_idx], WInv_[tw_idx], prime);
//     butt_intt_local(local[1], local[5], WInv[tw_idx], WInv_[tw_idx], prime);
//     butt_intt_local(local[2], local[6], WInv[tw_idx], WInv_[tw_idx], prime);
//     butt_intt_local(local[3], local[7], WInv[tw_idx], WInv_[tw_idx], prime);
//   }
//   for (int j = 0; j < 8; j++) {
//     if (local[j] >= prime)
//       local[j] -= prime;
//   }
//   N_init = t / 4 / radix * WarpID + Warp_t + pad * (t_idx / (radix * pad));
//   for (int j = 0; j < 8; j++) {
//     *(out_addr + N_init + t / 4 * j) = local[j];
//   }
// }
// }

// __global__ void Ntt8PointPerThreadPhase1(
//   const uint64_t* in_ptr,
//   uint64_t* out_ptr,
//   const int m,
//   const int num_prime,
//   const int N,
//   const int pad,
//   const int radix,
//   const uint64_t* base_inv,
//   const uint64_t* base_inv_,
//   const uint64_t* primes) {
// extern __shared__ uint64_t temp[];
// int Warp_t = threadIdx.x % pad;
// int WarpID = threadIdx.x / pad;
// for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (N / 8 * num_prime);
//      i += blockDim.x * gridDim.x) {
//   // size of a block
//   uint64_t local[8];
//   int t = N / 2 / m;
//   // prime idx
//   int np_idx = i / (N / 8);
//   // index in N/2 range
//   int N_idx = i % (N / 8);
//   // i'th block
//   int m_idx = N_idx / (t / 4);
//   int t_idx = N_idx % (t / 4);
//   // base address
//   const uint64_t* a_np = in_ptr + np_idx * N;
//   uint64_t* a_np_out = out_ptr + np_idx * N;
//   const uint64_t* prime_table = primes;
//   const uint64_t* W = base_inv + N * np_idx;
//   const uint64_t* W_ = base_inv_ + N * np_idx;
//   uint64_t prime = prime_table[np_idx];
//   int N_init = 2 * m_idx * t + t / 4 / radix * WarpID + Warp_t +
//       pad * (t_idx / (radix * pad));
//   for (int j = 0; j < 8; j++) {
//     local[j] = *(a_np + N_init + t / 4 * j);
//   }
//   __syncthreads();
//   int eradix = 8 * radix;
//   int tw_idx = m + m_idx;
//   for (int j = 0; j < 4; j++) {
//     butt_ntt_local(local[j], local[j + 4], W[tw_idx], W_[tw_idx], prime);
//   }
//   for (int j = 0; j < 2; j++) {
//     butt_ntt_local(
//         local[4 * j],
//         local[4 * j + 2],
//         W[2 * tw_idx + j],
//         W_[2 * tw_idx + j],
//         prime);
//     butt_ntt_local(
//         local[4 * j + 1],
//         local[4 * j + 3],
//         W[2 * tw_idx + j],
//         W_[2 * tw_idx + j],
//         prime);
//   }
//   for (int j = 0; j < 4; j++) {
//     butt_ntt_local(
//         local[2 * j],
//         local[2 * j + 1],
//         W[4 * tw_idx + j],
//         W_[4 * tw_idx + j],
//         prime);
//   }
//   for (int j = 0; j < 8; j++) {
//     temp[Warp_t * (eradix + pad) + WarpID + radix * j] = local[j];
//   }
//   int tail = 0;
//   __syncthreads();
// #pragma unroll
//   for (int j = 8, k = radix / 2; j < radix + 1; j *= 8, k >>= 3) {
//     int m_idx2 = WarpID / (k / 4);
//     int t_idx2 = WarpID % (k / 4);
//     for (int l = 0; l < 8; l++) {
//       local[l] = temp
//           [(eradix + pad) * Warp_t + 2 * m_idx2 * k + t_idx2 + (k / 4) * l];
//     }
//     int tw_idx2 = j * tw_idx + m_idx2;
//     for (int j2 = 0; j2 < 4; j2++) {
//       butt_ntt_local(
//           local[j2], local[j2 + 4], W[tw_idx2], W_[tw_idx2], prime);
//     }
//     for (int j2 = 0; j2 < 2; j2++) {
//       butt_ntt_local(
//           local[4 * j2],
//           local[4 * j2 + 2],
//           W[2 * tw_idx2 + j2],
//           W_[2 * tw_idx2 + j2],
//           prime);
//       butt_ntt_local(
//           local[4 * j2 + 1],
//           local[4 * j2 + 3],
//           W[2 * tw_idx2 + j2],
//           W_[2 * tw_idx2 + j2],
//           prime);
//     }
//     for (int j2 = 0; j2 < 4; j2++) {
//       butt_ntt_local(
//           local[2 * j2],
//           local[2 * j2 + 1],
//           W[4 * tw_idx2 + j2],
//           W_[4 * tw_idx2 + j2],
//           prime);
//     }

//     for (int l = 0; l < 8; l++) {
//       temp[(eradix + pad) * Warp_t + 2 * m_idx2 * k + t_idx2 + (k / 4) * l] =
//           local[l];
//     }
//     if (j == radix / 2)
//       tail = 1;
//     if (j == radix / 4)
//       tail = 2;
//     __syncthreads();
//   }
//   if (radix < 8)
//     tail = (radix == 4) ? 2 : 1;
//   if (tail == 1) {
//     for (int l = 0; l < 8; l++) {
//       local[l] = temp[(eradix + pad) * Warp_t + 8 * WarpID + l];
//     }
//     int tw_idx2 = (4 * radix) * tw_idx + 4 * WarpID;
//     butt_ntt_local(local[0], local[1], W[tw_idx2], W_[tw_idx2], prime);
//     butt_ntt_local(
//         local[2], local[3], W[tw_idx2 + 1], W_[tw_idx2 + 1], prime);
//     butt_ntt_local(
//         local[4], local[5], W[tw_idx2 + 2], W_[tw_idx2 + 2], prime);
//     butt_ntt_local(
//         local[6], local[7], W[tw_idx2 + 3], W_[tw_idx2 + 3], prime);
//     for (int l = 0; l < 8; l++) {
//       temp[(eradix + pad) * Warp_t + 8 * WarpID + l] = local[l];
//     }
//   } else if (tail == 2) {
//     for (int l = 0; l < 8; l++) {
//       local[l] = temp[(eradix + pad) * Warp_t + 8 * WarpID + l];
//     }
//     int tw_idx2 = 2 * radix * tw_idx + 2 * WarpID;
//     butt_ntt_local(local[0], local[2], W[tw_idx2], W_[tw_idx2], prime);
//     butt_ntt_local(local[1], local[3], W[tw_idx2], W_[tw_idx2], prime);
//     butt_ntt_local(
//         local[4], local[6], W[tw_idx2 + 1], W_[tw_idx2 + 1], prime);
//     butt_ntt_local(
//         local[5], local[7], W[tw_idx2 + 1], W_[tw_idx2 + 1], prime);
//     butt_ntt_local(
//         local[0], local[1], W[2 * tw_idx2], W_[2 * tw_idx2], prime);
//     butt_ntt_local(
//         local[2], local[3], W[2 * tw_idx2 + 1], W_[2 * tw_idx2 + 1], prime);
//     butt_ntt_local(
//         local[4], local[5], W[2 * tw_idx2 + 2], W_[2 * tw_idx2 + 2], prime);
//     butt_ntt_local(
//         local[6], local[7], W[2 * tw_idx2 + 3], W_[2 * tw_idx2 + 3], prime);
//     for (int l = 0; l < 8; l++) {
//       temp[(eradix + pad) * Warp_t + 8 * WarpID + l] = local[l];
//     }
//   }
//   __syncthreads();
//   for (int j = 0; j < 8; j++) {
//     local[j] = temp[Warp_t * (eradix + pad) + WarpID + radix * j];
//   }
//   for (int j = 0; j < 8; j++) {
//     *(a_np_out + N_init + t / 4 * j) = local[j];
//   }
// }
// }

// __global__ void Ntt8PointPerThreadPhase2(
//   const uint64_t* in_ptr,
//   uint64_t* out_ptr,
//   const int m,
//   const int num_prime,
//   const int N,
//   const int radix,
//   const uint64_t* base_inv,
//   const uint64_t* base_inv_,
//   const uint64_t* primes) {
// extern __shared__ uint64_t temp[];
// int set = threadIdx.x / radix;
// for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (N / 8 * num_prime);
//      i += blockDim.x * gridDim.x) {
//   // size of a block
//   uint64_t local[8];
//   int t = N / 2 / m;
//   // prime idx
//   int np_idx = num_prime - 1 - (i / (N / 8));
//   // index in N/2 range
//   int N_idx = i % (N / 8);
//   // i'th block
//   int m_idx = N_idx / (t / 4);
//   int t_idx = N_idx % (t / 4);
//   // base address
//   const uint64_t* a_np = in_ptr + np_idx * N;
//   uint64_t* a_np_out = out_ptr + np_idx * N;
//   const uint64_t* prime_table = primes;
//   uint64_t prime = prime_table[np_idx];
//   int N_init = 2 * m_idx * t + t_idx;
//   for (int j = 0; j < 8; j++) {
//     local[j] = *(a_np + N_init + t / 4 * j);
//   }
//   int tw_idx = m + m_idx;
//   const uint64_t* W = base_inv + N * np_idx;
//   const uint64_t* W_ = base_inv_ + N * np_idx;
//   for (int j = 0; j < 4; j++) {
//     butt_ntt_local(local[j], local[j + 4], W[tw_idx], W_[tw_idx], prime);
//   }
//   for (int j = 0; j < 2; j++) {
//     butt_ntt_local(
//         local[4 * j],
//         local[4 * j + 2],
//         W[2 * tw_idx + j],
//         W_[2 * tw_idx + j],
//         prime);
//     butt_ntt_local(
//         local[4 * j + 1],
//         local[4 * j + 3],
//         W[2 * tw_idx + j],
//         W_[2 * tw_idx + j],
//         prime);
//   }
//   for (int j = 0; j < 4; j++) {
//     butt_ntt_local(
//         local[2 * j],
//         local[2 * j + 1],
//         W[4 * tw_idx + j],
//         W_[4 * tw_idx + j],
//         prime);
//   }
//   for (int j = 0; j < 8; j++) {
//     temp[set * 8 * radix + t_idx + t / 4 * j] = local[j];
//   }
//   int tail = 0;
//   __syncthreads();
// #pragma unroll
//   for (int j = 8, k = t / 8; j < t / 4 + 1; j *= 8, k >>= 3) {
//     int m_idx2 = t_idx / (k / 4);
//     int t_idx2 = t_idx % (k / 4);
//     for (int l = 0; l < 8; l++) {
//       local[l] =
//           temp[set * 8 * radix + 2 * m_idx2 * k + t_idx2 + (k / 4) * l];
//     }
//     int tw_idx2 = j * tw_idx + m_idx2;
//     for (int j2 = 0; j2 < 4; j2++) {
//       butt_ntt_local(
//           local[j2], local[j2 + 4], W[tw_idx2], W_[tw_idx2], prime);
//     }
//     for (int j2 = 0; j2 < 2; j2++) {
//       butt_ntt_local(
//           local[4 * j2],
//           local[4 * j2 + 2],
//           W[2 * tw_idx2 + j2],
//           W_[2 * tw_idx2 + j2],
//           prime);
//       butt_ntt_local(
//           local[4 * j2 + 1],
//           local[4 * j2 + 3],
//           W[2 * tw_idx2 + j2],
//           W_[2 * tw_idx2 + j2],
//           prime);
//     }
//     for (int j2 = 0; j2 < 4; j2++) {
//       butt_ntt_local(
//           local[2 * j2],
//           local[2 * j2 + 1],
//           W[4 * tw_idx2 + j2],
//           W_[4 * tw_idx2 + j2],
//           prime);
//     }

//     for (int l = 0; l < 8; l++) {
//       temp[set * 8 * radix + 2 * m_idx2 * k + t_idx2 + (k / 4) * l] =
//           local[l];
//     }
//     if (j == t / 8)
//       tail = 1;
//     if (j == t / 16)
//       tail = 2;
//     __syncthreads();
//   }
//   if (tail == 1) {
//     for (int l = 0; l < 8; l++) {
//       local[l] = temp[set * 8 * radix + 8 * t_idx + l];
//     }
//     int tw_idx2 = t * tw_idx + 4 * t_idx;
//     butt_ntt_local(local[0], local[1], W[tw_idx2], W_[tw_idx2], prime);
//     butt_ntt_local(
//         local[2], local[3], W[tw_idx2 + 1], W_[tw_idx2 + 1], prime);
//     butt_ntt_local(
//         local[4], local[5], W[tw_idx2 + 2], W_[tw_idx2 + 2], prime);
//     butt_ntt_local(
//         local[6], local[7], W[tw_idx2 + 3], W_[tw_idx2 + 3], prime);
//     for (int l = 0; l < 8; l++) {
//       temp[set * 8 * radix + 8 * t_idx + l] = local[l];
//     }
//   } else if (tail == 2) {
//     for (int l = 0; l < 8; l++) {
//       local[l] = temp[set * 8 * radix + 8 * t_idx + l];
//     }
//     int tw_idx2 = (t / 2) * tw_idx + 2 * t_idx;
//     butt_ntt_local(local[0], local[2], W[tw_idx2], W_[tw_idx2], prime);
//     butt_ntt_local(local[1], local[3], W[tw_idx2], W_[tw_idx2], prime);
//     butt_ntt_local(
//         local[4], local[6], W[tw_idx2 + 1], W_[tw_idx2 + 1], prime);
//     butt_ntt_local(
//         local[5], local[7], W[tw_idx2 + 1], W_[tw_idx2 + 1], prime);
//     butt_ntt_local(
//         local[0], local[1], W[2 * tw_idx2], W_[2 * tw_idx2], prime);
//     butt_ntt_local(
//         local[2], local[3], W[2 * tw_idx2 + 1], W_[2 * tw_idx2 + 1], prime);
//     butt_ntt_local(
//         local[4], local[5], W[2 * tw_idx2 + 2], W_[2 * tw_idx2 + 2], prime);
//     butt_ntt_local(
//         local[6], local[7], W[2 * tw_idx2 + 3], W_[2 * tw_idx2 + 3], prime);
//     for (int l = 0; l < 8; l++) {
//       temp[set * 8 * radix + 8 * t_idx + l] = local[l];
//     }
//   }
//   __syncthreads();
//   for (int j = 0; j < 8; j++) {
//     local[j] = temp[set * 8 * radix + t_idx + t / 4 * j];
//     for (int k = 0; k < 3; k++) {
//       if (local[j] >= prime)
//         local[j] -= prime;
//     }
//   }
//   for (int j = 0; j < 8; j++) {
//     *(a_np_out + N_init + t / 4 * j) = local[j];
//   }
// }
// }

// __global__ void Ntt8PointPerThreadPhase1ExcludeSomeRange(
//   uint64_t* op,
//   const int m,
//   const int num_prime,
//   const int N,
//   const int start_prime_idx,
//   const int excluded_range_start,
//   const int excluded_range_end,
//   const int curr_limbs,
//   const int pad,
//   const int radix,
//   const uint64_t* base_inv,
//   const uint64_t* base_inv_,
//   const uint64_t* primes) {
// extern __shared__ uint64_t temp[];
// int Warp_t = threadIdx.x % pad;
// int WarpID = threadIdx.x / pad;
// // for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (N / 8 *
// // num_prime); i += blockDim.x * gridDim.x) { int i = blockIdx.x * blockDim.x
// // + threadIdx.x; size of a block
// uint64_t local[8];
// int t = N / 2 / m;
// // prime idx
// int np_idx = blockIdx.y + start_prime_idx;
// if (np_idx >= excluded_range_start && np_idx < excluded_range_end)
//   return;
// int prime_idx = np_idx;
// // index in N/2 range
// int N_idx = blockIdx.x * blockDim.x + threadIdx.x;
// // i'th block
// int m_idx = N_idx / (t / 4);
// int t_idx = N_idx % (t / 4);
// // base address
// uint64_t* a_np = op + np_idx * N;
// const uint64_t* prime_table = primes;
// const uint64_t* W = base_inv + N * prime_idx;
// const uint64_t* W_ = base_inv_ + N * prime_idx;
// uint64_t prime = prime_table[prime_idx];
// int N_init = 2 * m_idx * t + t / 4 / radix * WarpID + Warp_t +
//     pad * (t_idx / (radix * pad));
// for (int j = 0; j < 8; j++) {
//   local[j] = *(a_np + N_init + t / 4 * j);
// }
// __syncthreads();
// int eradix = 8 * radix;
// int tw_idx = m + m_idx;
// for (int j = 0; j < 4; j++) {
//   butt_ntt_local(local[j], local[j + 4], W[tw_idx], W_[tw_idx], prime);
// }
// for (int j = 0; j < 2; j++) {
//   butt_ntt_local(
//       local[4 * j],
//       local[4 * j + 2],
//       W[2 * tw_idx + j],
//       W_[2 * tw_idx + j],
//       prime);
//   butt_ntt_local(
//       local[4 * j + 1],
//       local[4 * j + 3],
//       W[2 * tw_idx + j],
//       W_[2 * tw_idx + j],
//       prime);
// }
// for (int j = 0; j < 4; j++) {
//   butt_ntt_local(
//       local[2 * j],
//       local[2 * j + 1],
//       W[4 * tw_idx + j],
//       W_[4 * tw_idx + j],
//       prime);
// }
// for (int j = 0; j < 8; j++) {
//   temp[Warp_t * (eradix + pad) + WarpID + radix * j] = local[j];
// }
// int tail = 0;
// __syncthreads();
// #pragma unroll
// for (int j = 8, k = radix / 2; j < radix + 1; j *= 8, k >>= 3) {
//   int m_idx2 = WarpID / (k / 4);
//   int t_idx2 = WarpID % (k / 4);
//   for (int l = 0; l < 8; l++) {
//     local[l] =
//         temp[(eradix + pad) * Warp_t + 2 * m_idx2 * k + t_idx2 + (k / 4) *
//         l];
//   }
//   int tw_idx2 = j * tw_idx + m_idx2;
//   for (int j2 = 0; j2 < 4; j2++) {
//     butt_ntt_local(local[j2], local[j2 + 4], W[tw_idx2], W_[tw_idx2], prime);
//   }
//   for (int j2 = 0; j2 < 2; j2++) {
//     butt_ntt_local(
//         local[4 * j2],
//         local[4 * j2 + 2],
//         W[2 * tw_idx2 + j2],
//         W_[2 * tw_idx2 + j2],
//         prime);
//     butt_ntt_local(
//         local[4 * j2 + 1],
//         local[4 * j2 + 3],
//         W[2 * tw_idx2 + j2],
//         W_[2 * tw_idx2 + j2],
//         prime);
//   }
//   for (int j2 = 0; j2 < 4; j2++) {
//     butt_ntt_local(
//         local[2 * j2],
//         local[2 * j2 + 1],
//         W[4 * tw_idx2 + j2],
//         W_[4 * tw_idx2 + j2],
//         prime);
//   }

//   for (int l = 0; l < 8; l++) {
//     temp[(eradix + pad) * Warp_t + 2 * m_idx2 * k + t_idx2 + (k / 4) * l] =
//         local[l];
//   }
//   if (j == radix / 2)
//     tail = 1;
//   if (j == radix / 4)
//     tail = 2;
//   __syncthreads();
// }
// if (radix < 8)
//   tail = (radix == 4) ? 2 : 1;
// if (tail == 1) {
//   for (int l = 0; l < 8; l++) {
//     local[l] = temp[(eradix + pad) * Warp_t + 8 * WarpID + l];
//   }
//   int tw_idx2 = (4 * radix) * tw_idx + 4 * WarpID;
//   butt_ntt_local(local[0], local[1], W[tw_idx2], W_[tw_idx2], prime);
//   butt_ntt_local(local[2], local[3], W[tw_idx2 + 1], W_[tw_idx2 + 1], prime);
//   butt_ntt_local(local[4], local[5], W[tw_idx2 + 2], W_[tw_idx2 + 2], prime);
//   butt_ntt_local(local[6], local[7], W[tw_idx2 + 3], W_[tw_idx2 + 3], prime);
//   for (int l = 0; l < 8; l++) {
//     temp[(eradix + pad) * Warp_t + 8 * WarpID + l] = local[l];
//   }
// } else if (tail == 2) {
//   for (int l = 0; l < 8; l++) {
//     local[l] = temp[(eradix + pad) * Warp_t + 8 * WarpID + l];
//   }
//   int tw_idx2 = 2 * radix * tw_idx + 2 * WarpID;
//   butt_ntt_local(local[0], local[2], W[tw_idx2], W_[tw_idx2], prime);
//   butt_ntt_local(local[1], local[3], W[tw_idx2], W_[tw_idx2], prime);
//   butt_ntt_local(local[4], local[6], W[tw_idx2 + 1], W_[tw_idx2 + 1], prime);
//   butt_ntt_local(local[5], local[7], W[tw_idx2 + 1], W_[tw_idx2 + 1], prime);
//   butt_ntt_local(local[0], local[1], W[2 * tw_idx2], W_[2 * tw_idx2], prime);
//   butt_ntt_local(
//       local[2], local[3], W[2 * tw_idx2 + 1], W_[2 * tw_idx2 + 1], prime);
//   butt_ntt_local(
//       local[4], local[5], W[2 * tw_idx2 + 2], W_[2 * tw_idx2 + 2], prime);
//   butt_ntt_local(
//       local[6], local[7], W[2 * tw_idx2 + 3], W_[2 * tw_idx2 + 3], prime);
//   for (int l = 0; l < 8; l++) {
//     temp[(eradix + pad) * Warp_t + 8 * WarpID + l] = local[l];
//   }
// }
// __syncthreads();
// for (int j = 0; j < 8; j++) {
//   local[j] = temp[Warp_t * (eradix + pad) + WarpID + radix * j];
// }
// for (int j = 0; j < 8; j++) {
//   *(a_np + N_init + t / 4 * j) = local[j];
// }
// }

// __global__ void Ntt8PointPerThreadPhase2ExcludeSomeRange(
//   uint64_t* op,
//   const int m,
//   const int num_prime,
//   const int N,
//   const int start_prime_idx,
//   const int excluded_range_start,
//   const int excluded_range_end,
//   const int curr_limbs,
//   const int radix,
//   const uint64_t* base_inv,
//   const uint64_t* base_inv_,
//   const uint64_t* primes) {
// extern __shared__ uint64_t temp[];
// int set = threadIdx.x / radix;
// // for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (N / 8 *
// // num_prime);
// //      i += blockDim.x * gridDim.x) {
// // size of a block
// uint64_t local[8];
// int t = N / 2 / m;
// // prime idx
// int np_idx = num_prime - 1 - blockIdx.y + start_prime_idx;
// if (np_idx >= excluded_range_start && np_idx < excluded_range_end)
//   return;
// int prime_idx = np_idx;
// // index in N/2 range
// int N_idx = blockIdx.x * blockDim.x + threadIdx.x;
// // i'th block
// int m_idx = N_idx / (t / 4);
// int t_idx = N_idx % (t / 4);
// // base address
// uint64_t* a_np = op + np_idx * N;
// const uint64_t* prime_table = primes;
// uint64_t prime = prime_table[prime_idx];
// int N_init = 2 * m_idx * t + t_idx;
// for (int j = 0; j < 8; j++) {
//   local[j] = *(a_np + N_init + t / 4 * j);
// }
// int tw_idx = m + m_idx;
// const uint64_t* W = base_inv + N * prime_idx;
// const uint64_t* W_ = base_inv_ + N * prime_idx;
// for (int j = 0; j < 4; j++) {
//   butt_ntt_local(local[j], local[j + 4], W[tw_idx], W_[tw_idx], prime);
// }
// for (int j = 0; j < 2; j++) {
//   butt_ntt_local(
//       local[4 * j],
//       local[4 * j + 2],
//       W[2 * tw_idx + j],
//       W_[2 * tw_idx + j],
//       prime);
//   butt_ntt_local(
//       local[4 * j + 1],
//       local[4 * j + 3],
//       W[2 * tw_idx + j],
//       W_[2 * tw_idx + j],
//       prime);
// }
// for (int j = 0; j < 4; j++) {
//   butt_ntt_local(
//       local[2 * j],
//       local[2 * j + 1],
//       W[4 * tw_idx + j],
//       W_[4 * tw_idx + j],
//       prime);
// }
// for (int j = 0; j < 8; j++) {
//   temp[set * 8 * radix + t_idx + t / 4 * j] = local[j];
// }
// int tail = 0;
// __syncthreads();
// #pragma unroll
// for (int j = 8, k = t / 8; j < t / 4 + 1; j *= 8, k >>= 3) {
//   int m_idx2 = t_idx / (k / 4);
//   int t_idx2 = t_idx % (k / 4);
//   for (int l = 0; l < 8; l++) {
//     local[l] = temp[set * 8 * radix + 2 * m_idx2 * k + t_idx2 + (k / 4) * l];
//   }
//   int tw_idx2 = j * tw_idx + m_idx2;
//   for (int j2 = 0; j2 < 4; j2++) {
//     butt_ntt_local(local[j2], local[j2 + 4], W[tw_idx2], W_[tw_idx2], prime);
//   }
//   for (int j2 = 0; j2 < 2; j2++) {
//     butt_ntt_local(
//         local[4 * j2],
//         local[4 * j2 + 2],
//         W[2 * tw_idx2 + j2],
//         W_[2 * tw_idx2 + j2],
//         prime);
//     butt_ntt_local(
//         local[4 * j2 + 1],
//         local[4 * j2 + 3],
//         W[2 * tw_idx2 + j2],
//         W_[2 * tw_idx2 + j2],
//         prime);
//   }
//   for (int j2 = 0; j2 < 4; j2++) {
//     butt_ntt_local(
//         local[2 * j2],
//         local[2 * j2 + 1],
//         W[4 * tw_idx2 + j2],
//         W_[4 * tw_idx2 + j2],
//         prime);
//   }

//   for (int l = 0; l < 8; l++) {
//     temp[set * 8 * radix + 2 * m_idx2 * k + t_idx2 + (k / 4) * l] = local[l];
//   }
//   if (j == t / 8)
//     tail = 1;
//   if (j == t / 16)
//     tail = 2;
//   __syncthreads();
// }
// if (tail == 1) {
//   for (int l = 0; l < 8; l++) {
//     local[l] = temp[set * 8 * radix + 8 * t_idx + l];
//   }
//   int tw_idx2 = t * tw_idx + 4 * t_idx;
//   butt_ntt_local(local[0], local[1], W[tw_idx2], W_[tw_idx2], prime);
//   butt_ntt_local(local[2], local[3], W[tw_idx2 + 1], W_[tw_idx2 + 1], prime);
//   butt_ntt_local(local[4], local[5], W[tw_idx2 + 2], W_[tw_idx2 + 2], prime);
//   butt_ntt_local(local[6], local[7], W[tw_idx2 + 3], W_[tw_idx2 + 3], prime);
//   for (int l = 0; l < 8; l++) {
//     temp[set * 8 * radix + 8 * t_idx + l] = local[l];
//   }
// } else if (tail == 2) {
//   for (int l = 0; l < 8; l++) {
//     local[l] = temp[set * 8 * radix + 8 * t_idx + l];
//   }
//   int tw_idx2 = (t / 2) * tw_idx + 2 * t_idx;
//   butt_ntt_local(local[0], local[2], W[tw_idx2], W_[tw_idx2], prime);
//   butt_ntt_local(local[1], local[3], W[tw_idx2], W_[tw_idx2], prime);
//   butt_ntt_local(local[4], local[6], W[tw_idx2 + 1], W_[tw_idx2 + 1], prime);
//   butt_ntt_local(local[5], local[7], W[tw_idx2 + 1], W_[tw_idx2 + 1], prime);
//   butt_ntt_local(local[0], local[1], W[2 * tw_idx2], W_[2 * tw_idx2], prime);
//   butt_ntt_local(
//       local[2], local[3], W[2 * tw_idx2 + 1], W_[2 * tw_idx2 + 1], prime);
//   butt_ntt_local(
//       local[4], local[5], W[2 * tw_idx2 + 2], W_[2 * tw_idx2 + 2], prime);
//   butt_ntt_local(
//       local[6], local[7], W[2 * tw_idx2 + 3], W_[2 * tw_idx2 + 3], prime);
//   for (int l = 0; l < 8; l++) {
//     temp[set * 8 * radix + 8 * t_idx + l] = local[l];
//   }
// }
// __syncthreads();
// for (int j = 0; j < 8; j++) {
//   local[j] = temp[set * 8 * radix + t_idx + t / 4 * j];
//   for (int k = 0; k < 3; k++) {
//     if (local[j] >= prime)
//       local[j] -= prime;
//   }
// }
// for (int j = 0; j < 8; j++) {
//   *(a_np + N_init + t / 4 * j) = local[j];
// }
// // }
// }
