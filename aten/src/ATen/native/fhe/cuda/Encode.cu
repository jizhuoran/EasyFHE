#include <ATen/Dispatch_v2.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/interned_strings.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/thread_constants.h>
#include <ATen/ops/copy.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>
#include "ATen/native/fhe/cuda/CommonOperation.h"
#include "ATen/native/fhe/cuda/Utils.cuh"

#define MAX_64BIT_VALUE 9223372036854775295LL
#define MAX_BITS_IN_WORD 61

namespace fhe {
static constexpr int kEncodeBlockSize = 256;

template <typename DDTYPE>
__global__ void new_fit_to_native_vector_kernel(
    DDTYPE* in_ptr,
    double scaling_factor,
    int64_t bigValueHf,
    uint64_t* out_ptr,
    uint64_t* native_modulus,
    uint64_t* max_int_diffs_ptr,
    uint64_t* barret_ratio_ptr,
    uint64_t* barret_k_ptr,
    int64_t N,
    const size_t L_OUTN,
    const size_t L_INN,
    int64_t slots,
    int64_t gap) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  auto cipher_id = blockIdx.z;
  in_ptr += cipher_id * L_INN;
  out_ptr += cipher_id * L_OUTN;

  if (i < slots) {
    const int l = blockIdx.y;
    int64_t diff = max_int_diffs_ptr[l];
    // const auto n = in_ptr[i];
    int64_t re = llround(in_ptr[2 * i] * scaling_factor);
    int64_t im = llround(in_ptr[2 * i + 1] * scaling_factor);

    re = (re < 0) ? (MAX_64BIT_VALUE + re) : re;
    im = (im < 0) ? (MAX_64BIT_VALUE + im) : im;

    uint64_t re_ = re;
    uint64_t im_ = im;

    barret_reduction_64_64(
        re_, re_, native_modulus[l], barret_ratio_ptr[l], barret_k_ptr[l]);
    barret_reduction_64_64(
        im_, im_, native_modulus[l], barret_ratio_ptr[l], barret_k_ptr[l]);

    if (re > bigValueHf) {
      re_ = sub_mod(re_, diff, native_modulus[l]);
    }
    if (im > bigValueHf) {
      im_ = sub_mod(im_, diff, native_modulus[l]);
    }

    out_ptr[l * N + gap * i] = re_;
    out_ptr[l * N + gap * (i + slots)] = im_;
  }
}

} // namespace fhe

namespace at::native {

static void encode_template(
    Tensor& res,
    const Tensor& input,
    const Tensor& max_int_diffs,
    int64_t N,
    int64_t cur_limbs,
    int64_t slots,
    double scaling_factor,
    bool is_ext,
    int64_t sizeP,
    const Tensor& primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    const Tensor& power_of_roots_shoup,
    const Tensor& power_of_roots) {
  auto res_ptr_ = reinterpret_cast<uint64_t*>(res.data_ptr<uint64_t>());
  auto primes_ptr = reinterpret_cast<uint64_t*>(primes.data_ptr<uint64_t>());
  auto max_int_diffs_ptr =
      reinterpret_cast<uint64_t*>(max_int_diffs.data_ptr<uint64_t>());
  auto barret_ratio_ptr =
      reinterpret_cast<uint64_t*>(barret_ratio.data_ptr<uint64_t>());
  auto barret_k_ptr =
      reinterpret_cast<uint64_t*>(barret_k.data_ptr<uint64_t>());

  auto num_plaintext = input.size(0);

  auto L_IN = 1;
  auto L_INN = L_IN * input.sizes()[1];

  auto L_OUT = res.sizes()[1];
  auto L_OUTN = L_OUT * res.sizes()[2];

  AT_DISPATCH_V2(
      input.scalar_type(),
      "encode_impl",
      AT_WRAP([&]() {
        auto input_ptr_ =
            reinterpret_cast<scalar_t*>(input.data_ptr<scalar_t>());
        auto gap = N / (slots * 2);
        auto stream = at::cuda::getCurrentCUDAStream();
        dim3 block(fhe::kEncodeBlockSize);
        dim3 grid(
            (slots + block.x - 1) / block.x,
            cur_limbs + (is_ext ? sizeP : 0),
            num_plaintext);
        fhe::new_fit_to_native_vector_kernel<<<grid, block, 0, stream>>>(
            input_ptr_,
            scaling_factor,
            MAX_64BIT_VALUE >> 1,
            res_ptr_,
            primes_ptr,
            max_int_diffs_ptr,
            barret_ratio_ptr,
            barret_k_ptr,
            N,
            L_OUTN,
            L_INN,
            slots,
            gap);
      }),
      kFloat,
      kDouble);

  NTT_impl(
      res_ptr_,
      cur_limbs,
      N,
      L_OUT,
      1,
      num_plaintext,
      primes.data_ptr<uint64_t>(),
      power_of_roots_shoup.data_ptr<uint64_t>(),
      power_of_roots.data_ptr<uint64_t>());

  if (is_ext) {
    auto L = power_of_roots.numel() / N - sizeP;
    NTT_impl(
        res_ptr_ + cur_limbs * N,
        sizeP,
        N,
        L_OUT,
        1,
        num_plaintext,
        primes.data_ptr<uint64_t>() + cur_limbs,
        power_of_roots_shoup.data_ptr<uint64_t>() + L * N,
        power_of_roots.data_ptr<uint64_t>() + L * N);
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

Tensor encode_cuda(
    const Tensor& inverse_internal,
    int64_t N,
    int64_t cur_limbs,
    int64_t slots,
    double scaling_factor,
    bool is_ext,
    int64_t sizeP,
    const Tensor& primes,
    const Tensor& max_int_diffs,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    const Tensor& power_of_roots_shoup,
    const Tensor& power_of_roots) {
  TORCH_INTERNAL_ASSERT(inverse_internal.dim() == 2);
  auto num_plaintext = inverse_internal.size(0);
  Tensor out = at::zeros(
      {num_plaintext, cur_limbs + (is_ext ? sizeP : 0), N}, primes.options());

  // for (size_t i = 0; i < num_plaintext; ++i) {
  //   auto inverse_internal_view = inverse_internal[i];
  //   auto out_view = out[i];
  encode_template(
      out,
      inverse_internal,
      max_int_diffs,
      N,
      cur_limbs,
      slots,
      scaling_factor,
      is_ext,
      sizeP,
      primes,
      barret_ratio,
      barret_k,
      power_of_roots_shoup,
      power_of_roots);
  // }

  return out;
}

} // namespace at::native
