#pragma once

#include <cstddef>

namespace fhe {

inline constexpr size_t kWorkPerThread = 1;
inline constexpr size_t kWarpSize = 32;
inline constexpr size_t kNumWarps = 8;
inline constexpr size_t kBlockSize = kWarpSize * kNumWarps;
inline constexpr size_t kWorkPerBlock = kWorkPerThread * kBlockSize;

constexpr size_t ceil_div(size_t value, size_t divisor) {
  return (value + divisor - 1) / divisor;
}

constexpr size_t launch_blocks(size_t work_items) {
  return ceil_div(work_items, kWorkPerBlock);
}

} // namespace fhe

#define WORK_PER_THREAD (::fhe::kWorkPerThread)
#define WARP_SIZE (::fhe::kWarpSize)
#define NUM_WARPS (::fhe::kNumWarps)
#define BLOCK_SIZE (::fhe::kBlockSize)
#define WORK_PER_BLOCK (::fhe::kWorkPerBlock)
#define num_blocks(n) (::fhe::launch_blocks(static_cast<size_t>(n)))

// clang-format off
#define DISPATCH_BATCH_FUNC(CASE, DISPATCH_FUNC)                              \
  switch (CASE) {                                                             \
    DISPATCH_FUNC(1)   DISPATCH_FUNC(2)   DISPATCH_FUNC(3)   DISPATCH_FUNC(4) \
    DISPATCH_FUNC(5)   DISPATCH_FUNC(6)   DISPATCH_FUNC(7)   DISPATCH_FUNC(8) \
    DISPATCH_FUNC(9)   DISPATCH_FUNC(10)  DISPATCH_FUNC(11)  DISPATCH_FUNC(12)\
    DISPATCH_FUNC(13)  DISPATCH_FUNC(14)  DISPATCH_FUNC(15)  DISPATCH_FUNC(16)\
    DISPATCH_FUNC(17)  DISPATCH_FUNC(18)  DISPATCH_FUNC(19)  DISPATCH_FUNC(20)\
    DISPATCH_FUNC(21)  DISPATCH_FUNC(22)  DISPATCH_FUNC(23)  DISPATCH_FUNC(24)\
    DISPATCH_FUNC(25)  DISPATCH_FUNC(26)  DISPATCH_FUNC(27)  DISPATCH_FUNC(28)\
    DISPATCH_FUNC(29)  DISPATCH_FUNC(30)  DISPATCH_FUNC(31)  DISPATCH_FUNC(32)\
    default:                                                                  \
      AT_ERROR("Unsupported batch size");                                     \
  }
// clang-format on
