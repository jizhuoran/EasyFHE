#include <ATen/Dispatch_v2.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/interned_strings.h>
#include <ATen/ops/copy.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/stack.h>
#include <ATen/ops/zeros.h>
#include <omp.h>

#include "ATen/native/fhe/cpu/Utils.h"

#pragma clang diagnostic ignored "-Wmissing-prototypes"

namespace fhe {

template <size_t NUM_CV>
void automorphism_transform_kernel(
    uint64_t* out,
    const uint64_t* in,
    size_t l,
    size_t N,
    size_t LN,
    size_t BLN,
    size_t num_batch,
    const int* precomp_vec) {
  const int max_threads = omp_get_max_threads();
#pragma omp parallel for collapse(3) schedule(static) num_threads(max_threads)
  for (size_t batch_id = 0; batch_id < num_batch; ++batch_id) {
    for (size_t i = 0; i < l; ++i) {
      for (size_t tid = 0; tid < N; ++tid) {
        const auto precomp_index = static_cast<size_t>(precomp_vec[tid]);
        for (size_t cv = 0; cv < NUM_CV; ++cv) {
          out[cv * BLN + batch_id * LN + i * N + tid] =
              in[cv * BLN + batch_id * LN + i * N + precomp_index];
        }
      }
    }
  }
}

} // namespace fhe

namespace at::native {

static void automorphism_transform_template(
    Tensor& out,
    const Tensor& in,
    int64_t l,
    int64_t N,
    const Tensor& precomp_vec) {
  auto* out_ptr = out.data_ptr<uint64_t>();
  auto* in_ptr = in.data_ptr<uint64_t>();
  auto* precomp_vec_ptr = precomp_vec.data_ptr<int32_t>();

  const auto LN = in.sizes()[2] * N;
  const auto BLN = in.sizes()[1] * LN;

  if (out.sizes()[0] == 1) {
    fhe::automorphism_transform_kernel<1>(
        out_ptr,
        in_ptr,
        static_cast<size_t>(l),
        static_cast<size_t>(N),
        static_cast<size_t>(LN),
        static_cast<size_t>(BLN),
        static_cast<size_t>(out.sizes()[1]),
        precomp_vec_ptr);
  } else if (out.sizes()[0] == 2) {
    fhe::automorphism_transform_kernel<2>(
        out_ptr,
        in_ptr,
        static_cast<size_t>(l),
        static_cast<size_t>(N),
        static_cast<size_t>(LN),
        static_cast<size_t>(BLN),
        static_cast<size_t>(out.sizes()[1]),
        precomp_vec_ptr);
  } else {
    TORCH_INTERNAL_ASSERT(false, "Unsupported number of cv");
  }
}

Tensor automorphism_transform_cpu(
    const Tensor& from,
    int64_t l,
    int64_t N,
    const Tensor& precomp_vec) {
  TORCH_INTERNAL_ASSERT(from.dim() == 4);
  Tensor out = at::empty_like(from);
  automorphism_transform_template(out, from, l, N, precomp_vec);
  return out;
}

} // namespace at::native
