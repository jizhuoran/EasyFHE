#include <ATen/Dispatch_v2.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/interned_strings.h>
#include <ATen/ops/copy.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>

namespace at::native {

Tensor pre_encode_cpu(
    const Tensor& input,
    int64_t slots,
    int64_t M,
    const Tensor& rotGroup,
    const Tensor& ksiPows,
    const Tensor& bitrev) {

  Tensor out = at::empty({slots}, input.options().dtype(kComplexDouble));
  Tensor workspace = at::zeros({slots}, input.options().dtype(kComplexDouble));
  
  auto input_ptr = input.data_ptr<double>();
  auto workspace_ptr = workspace.data_ptr<complex<double>>();

  for(int i = 0; i < input.numel(); ++i) {
    workspace_ptr[i] = complex<double>(input_ptr[i], 0.0);
  }
  auto out_ptr = out.data_ptr<complex<double>>();
  auto rotGroup_ptr = rotGroup.data_ptr<uint32_t>();
  auto ksiPows_ptr = ksiPows.data_ptr<complex<double>>();
  auto bitrev_ptr = bitrev.data_ptr<uint32_t>();

  auto vals_size = slots;
  auto len_size = vals_size;
  while (len_size >= 1) {
    auto len_h = len_size >> 1;
    auto len_q = len_size << 2;
    auto gap = M / len_q;

    for (size_t i = 0; i < vals_size; i += len_size) {
      for (size_t j = 0; j < len_h; ++j) {
        auto idx = (len_q - (rotGroup_ptr[j] % len_q)) * gap;
        auto u = workspace_ptr[i + j] + workspace_ptr[i + j + len_h];
        auto v = workspace_ptr[i + j] - workspace_ptr[i + j + len_h];
        v *= ksiPows_ptr[idx];
        workspace_ptr[i + j] = u;
        workspace_ptr[i + j + len_h] = v;
      }
    }
    len_size >>= 1;
  }

  for (size_t i = 0; i < vals_size; ++i) {
    out_ptr[i] = workspace_ptr[bitrev_ptr[i]] / vals_size;
  }

  return out;
}

} // namespace at::native
