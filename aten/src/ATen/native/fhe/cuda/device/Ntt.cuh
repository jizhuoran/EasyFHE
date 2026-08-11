#pragma once

#include <cstddef>
#include <cstdint>

#include "ATen/native/fhe/cuda/device/Launch.cuh"
#include "ATen/native/fhe/cuda/device/Modular.cuh"

namespace fhe::ntt {

__device__ __forceinline__ uint64_t reduce_lazy_4p(
    uint64_t value,
    uint64_t prime,
    uint64_t two_p) {
  if (value >= two_p) {
    value -= two_p;
  }
  if (value >= prime) {
    value -= prime;
  }
  return value;
}

__device__ __forceinline__ void butterfly(
    uint64_t& a,
    uint64_t& b,
    uint64_t root,
    uint64_t root_shoup,
    uint64_t prime,
    uint64_t two_p) {
  const uint64_t product = mul_and_reduce_shoup(b, root, root_shoup, prime);
  if (a >= two_p) {
    a -= two_p;
  }
  b = a + (two_p - product);
  a += product;
}

template <uint8_t Radix>
__device__ __forceinline__ void local_radix(
    uint64_t& local0,
    uint64_t& local1,
    uint64_t& local2,
    uint64_t& local3,
    uint64_t& local4,
    uint64_t& local5,
    uint64_t& local6,
    uint64_t& local7,
    uint32_t root_offset,
    const uint64_t* __restrict__ roots,
    const uint64_t* __restrict__ roots_shoup,
    uint64_t prime,
    uint64_t two_p) {
  static_assert(Radix == 2 || Radix == 4 || Radix == 8);
  if constexpr (Radix >= 8) {
    const uint64_t root = roots[root_offset];
    const uint64_t root_shoup = roots_shoup[root_offset];
    butterfly(local0, local4, root, root_shoup, prime, two_p);
    butterfly(local1, local5, root, root_shoup, prime, two_p);
    butterfly(local2, local6, root, root_shoup, prime, two_p);
    butterfly(local3, local7, root, root_shoup, prime, two_p);
  }

  if constexpr (Radix >= 4) {
    const uint32_t offset = 2 * root_offset;
    const uint64_t root0 = roots[offset];
    const uint64_t root0_shoup = roots_shoup[offset];
    const uint64_t root1 = roots[offset + 1];
    const uint64_t root1_shoup = roots_shoup[offset + 1];
    butterfly(local0, local2, root0, root0_shoup, prime, two_p);
    butterfly(local1, local3, root0, root0_shoup, prime, two_p);
    butterfly(local4, local6, root1, root1_shoup, prime, two_p);
    butterfly(local5, local7, root1, root1_shoup, prime, two_p);
  }

  if constexpr (Radix >= 2) {
    const uint32_t offset = 4 * root_offset;
    const uint64_t root0 = roots[offset];
    const uint64_t root0_shoup = roots_shoup[offset];
    const uint64_t root1 = roots[offset + 1];
    const uint64_t root1_shoup = roots_shoup[offset + 1];
    const uint64_t root2 = roots[offset + 2];
    const uint64_t root2_shoup = roots_shoup[offset + 2];
    const uint64_t root3 = roots[offset + 3];
    const uint64_t root3_shoup = roots_shoup[offset + 3];
    butterfly(local0, local1, root0, root0_shoup, prime, two_p);
    butterfly(local2, local3, root1, root1_shoup, prime, two_p);
    butterfly(local4, local5, root2, root2_shoup, prime, two_p);
    butterfly(local6, local7, root3, root3_shoup, prime, two_p);
  }
}

template <int NumRounds>
__device__ __forceinline__ void warp_butterfly(
    uint64_t& first,
    uint64_t& second,
    uint32_t& stage_offset,
    uint32_t lane_id,
    const uint64_t* __restrict__ roots,
    const uint64_t* __restrict__ roots_shoup,
    uint64_t prime,
    uint64_t two_p) {
  static_assert(NumRounds >= 2);
  butterfly(
      first,
      second,
      roots[stage_offset],
      roots_shoup[stage_offset],
      prime,
      two_p);

#pragma unroll
  for (int shift = NumRounds - 2; shift >= 0; --shift) {
    const uint32_t offset = 1u << shift;
    const bool lower_half = (lane_id & offset) == 0;
    auto exchanged = lower_half ? second : first;
    exchanged = __shfl_xor_sync(0xFFFFFFFF, exchanged, offset);
    if (lower_half) {
      second = exchanged;
    } else {
      first = exchanged;
    }

    stage_offset <<= 1;
    const uint32_t root_index = stage_offset + (lane_id >> shift);
    butterfly(
        first,
        second,
        roots[root_index],
        roots_shoup[root_index],
        prime,
        two_p);
  }
}

template <size_t LogN, int NumWarps>
__device__ __forceinline__ ulonglong2 phase2_pair(
    uint64_t* __restrict__ inout,
    uint32_t polynomial_index,
    uint32_t block_index,
    const uint64_t* __restrict__ roots,
    const uint64_t* __restrict__ roots_shoup,
    uint64_t prime,
    uint64_t two_p,
    uint64_t (&tile)[2][NumWarps][kWarpSize + 1]) {
  constexpr size_t kLogRadix = LogN - 6;
  constexpr int kRadix = 64;
  constexpr int kDataSize = 1u << kLogRadix;
  constexpr int kBlockSize = 1u << (kLogRadix - 1);
  auto rows = reinterpret_cast<uint64_t(*)[kRadix][kDataSize]>(inout);
  uint32_t stage_offset = kRadix + block_index;

  uint64_t first = rows[polynomial_index][block_index][threadIdx.x];
  uint64_t second =
      rows[polynomial_index][block_index][threadIdx.x + kBlockSize];

  tile[0][threadIdx.x % NumWarps][threadIdx.x / NumWarps] = first;
  tile[1][threadIdx.x % NumWarps][threadIdx.x / NumWarps] = second;
  __syncthreads();

  uint32_t lane_id = threadIdx.x % kWarpSize;
  uint32_t group_id = threadIdx.x / kWarpSize;
  first = tile[0][group_id][lane_id];
  second = tile[1][group_id][lane_id];

  warp_butterfly<6>(
      first,
      second,
      stage_offset,
      lane_id,
      roots,
      roots_shoup,
      prime,
      two_p);

  tile[0][group_id][lane_id] = first;
  tile[1][group_id][lane_id] = second;
  __syncthreads();

  constexpr int kSecondGroupSize = kDataSize / kWarpSize / 4;
  lane_id = threadIdx.x % kSecondGroupSize;
  group_id = threadIdx.x / kSecondGroupSize;

  const uint32_t half_group = group_id >> 1;
  if ((group_id & 1) == 0) {
    first = tile[0][lane_id][half_group];
    second = tile[0][lane_id + kSecondGroupSize][half_group];
  } else {
    first = tile[1][lane_id][half_group];
    second = tile[1][lane_id + kSecondGroupSize][half_group];
  }

  stage_offset = stage_offset * 2 + group_id;
  warp_butterfly<kLogRadix - 6>(
      first,
      second,
      stage_offset,
      lane_id,
      roots,
      roots_shoup,
      prime,
      two_p);

  first = reduce_lazy_4p(first, prime, two_p);
  second = reduce_lazy_4p(second, prime, two_p);
  return {first, second};
}

} // namespace fhe::ntt
