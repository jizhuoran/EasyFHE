#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/TensorOperators.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#endif

namespace at::native {

template <
    typename index_t,
    void compute(const index_t*, const int64_t*, index_t*, int64_t, int64_t)>
static inline Tensor repeat_interleave_common(
    const Tensor& repeats,
    std::optional<int64_t> output_size) {
  (void)repeats;
  (void)output_size;
  TORCH_CHECK(
      false,
      "repeat_interleave with tensor repeats is disabled in EasyFHE because cumsum is not supported");
  return Tensor();
}

} // namespace at::native
