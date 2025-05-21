#include <ATen/Dispatch_v2.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/interned_strings.h>
#include <ATen/ops/copy.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>
#include "ATen/native/fhe/cpu/CommonOperation.h"
#include "ATen/native/fhe/cpu/Utils.h"
#include <limits>
#define MAX_64BIT_VALUE 9223372036854775295LL


namespace fhe {
template <typename DDTYPE>
void new_fit_to_native_vector_host(
    DDTYPE* inverse,
    double scaling_factor,
    int64_t bigValueHf,
    uint64_t* native_vec,
    const uint64_t* native_modulus,
    const uint64_t* max_int_diffs_ptr,
    const uint64_t* barret_ratio_ptr,
    const uint64_t* barret_k_ptr,
    int64_t N,
    int64_t slots,
    int64_t gap,
    int64_t cur_limbs // 相当于 CUDA 中的 gridDim.y，即每个 slot 扫描多少 limbs
) {

    for (int64_t l = 0; l < cur_limbs; ++l) {
        uint64_t modulus = native_modulus[l];
        uint64_t diff = max_int_diffs_ptr[l];
        uint64_t ratio = barret_ratio_ptr[l];
        uint64_t k = barret_k_ptr[l];
        for (int64_t i = 0; i < slots; ++i) {
    int64_t re = static_cast<int64_t>(std::llround(inverse[2 * i] * scaling_factor));
    int64_t im = static_cast<int64_t>(std::llround(inverse[2 * i + 1] * scaling_factor));             
    if (re < 0) re = MAX_64BIT_VALUE + re;
    if (im < 0) im = MAX_64BIT_VALUE + im;
    uint64_t re_u = static_cast<uint64_t>(re);
    uint64_t im_u = static_cast<uint64_t>(im);
    re_u = barret_reduction_64_64(re_u, modulus, ratio, k);
    im_u = barret_reduction_64_64(im_u, modulus, ratio, k);

    if (re > bigValueHf) {
        re_u = sub_mod(re_u, diff, modulus);
    }
    if (im > bigValueHf) {
        im_u = sub_mod(im_u, diff, modulus);
    }
    native_vec[l * N + gap * i] = re_u;
    native_vec[l * N + gap * (i + slots)] = im_u;
            }
        }
}
    
}
namespace at::native {
static void encode_template_cpu(
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
    const Tensor& power_of_roots) 
{
    auto elements_ptr      = reinterpret_cast<uint64_t*>(res.data_ptr<uint64_t>());
    auto primes_ptr        = reinterpret_cast<uint64_t*>(primes.data_ptr<uint64_t>());
    auto max_int_diffs_ptr = reinterpret_cast<uint64_t*>(max_int_diffs.data_ptr<uint64_t>());
    auto barret_ratio_ptr  = reinterpret_cast<uint64_t*>(barret_ratio.data_ptr<uint64_t>());
    auto barret_k_ptr      = reinterpret_cast<uint64_t*>(barret_k.data_ptr<uint64_t>());
    const auto gap = N / (slots * 2);
   AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "encode_template_cpu", [&]() {
        auto input_ptr = input.data_ptr<scalar_t>();
        fhe::new_fit_to_native_vector_host<scalar_t>(
            input_ptr,
            scaling_factor,
            static_cast<int64_t>(MAX_64BIT_VALUE >> 1),
            elements_ptr,
            primes_ptr,
            max_int_diffs_ptr,
            barret_ratio_ptr,
            barret_k_ptr,
            N,
            slots,
            gap,
            cur_limbs);
    });
 NTT_impl(
    elements_ptr,
    0,
    cur_limbs,
    N,
    power_of_roots_shoup,
    primes,
    power_of_roots);
    if (is_ext) {
    const auto L = power_of_roots.numel() / N - sizeP;
    auto offset = L * N;
    
    // 创建偏移的张量视图
    Tensor power_of_roots_shoup_offset = power_of_roots_shoup.narrow(0, offset, power_of_roots_shoup.numel() - offset);
    Tensor power_of_roots_offset = power_of_roots.narrow(0, offset, power_of_roots.numel() - offset);
    Tensor primes_offset = primes.narrow(0, cur_limbs, primes.numel() - cur_limbs);
    NTT_impl(
        elements_ptr + cur_limbs * N, 
        0,                    
        sizeP,                         
        N,                            
        power_of_roots_shoup_offset,       
        primes,                        
        power_of_roots_offset);       
}
}

Tensor encode_cpu(
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
  
  Tensor out = at::zeros({cur_limbs + (is_ext ? sizeP : 0), N}, primes.options());
  encode_template_cpu(
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
  return out;
}
}