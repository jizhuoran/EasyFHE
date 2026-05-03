#include <ATen/Dispatch_v2.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/interned_strings.h>
#include <ATen/ops/copy.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>

#include "ATen/native/fhe/cpu/CommonOperation.h"

#pragma clang diagnostic ignored "-Wmissing-prototypes"

namespace at::native {

static void extend_ciphertext_template_precomp(
    Tensor& res,
    const Tensor& in,
    Tensor& temp,
    Tensor& ctxtDCRT_modq,
    int64_t N,
    int64_t L0,
    int64_t level,
    int64_t composite_degree,
    const Tensor& qj,                       // [cd]
    const Tensor& qj_psinv,                 // [cd]
    const Tensor& qhat_inv_modqj_mod,        // [cd, cd]
    const Tensor& qhat_inv_modqj_psinv,      // [cd, cd]
    const Tensor& qjProduct_mod,             // [L0]
    const Tensor& qjProduct_psinv,           // [L0]
    const Tensor& qjProductD_mod,            // [cd, L0]
    const Tensor& qjProductD_psinv,          // [cd, L0]
    const Tensor& moduliQ,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_power_of_roots) {

  auto op = in.clone();
  auto op_ptr = reinterpret_cast<uint64_t*>(op.data_ptr<uint64_t>());
  auto res_ptr = reinterpret_cast<uint64_t*>(res.data_ptr<uint64_t>());
  auto ctxt_ptr = reinterpret_cast<uint64_t*>(ctxtDCRT_modq.data_ptr<uint64_t>());
  auto temp_ptr = reinterpret_cast<uint64_t*>(temp.data_ptr<uint64_t>());

  uint32_t init_element_index = static_cast<uint32_t>(composite_degree);

  iNTT_impl(
      op_ptr,
      op_ptr,
      0,
      composite_degree,
      composite_degree,
      level,
      N,
      inverse_power_of_roots_div_two,
      moduliQ,
      inverse_scaled_power_of_roots_div_two);

  for (int64_t k = 0; k < composite_degree; k++) {
    Tensor op2k_mod = qhat_inv_modqj_mod.select(0, k).contiguous();     // [cd]
    Tensor op2k_psinv = qhat_inv_modqj_psinv.select(0, k).contiguous(); // [cd]

    const_mult_batch_(
        op_ptr,
        op2k_mod,
        op2k_psinv,
        0,
        composite_degree,
        0,
        0,
        N,
        ctxt_ptr + k * L0 * N,
        moduliQ);
  }

  switch_modulus(ctxt_ptr, temp_ptr, moduliQ, 0, L0, N);

  const_mult_batch_(
      temp_ptr,
      qjProduct_mod,
      qjProduct_psinv,
      0,
      L0,
      0,
      0,
      N,
      temp_ptr,
      moduliQ);

  for (int64_t d = 1; d < composite_degree; d++) {
    switch_modulus(
        ctxt_ptr + (d * L0 + d) * N,
        temp_ptr + (static_cast<int64_t>(init_element_index) * L0 * N),
        moduliQ,
        d,
        L0,
        N);

    uint64_t* dst_group_d = temp_ptr + d * L0 * N;

    if (d > 0) {
      const_mult_batch_(
          temp_ptr,
          qj,
          qj_psinv,
          0,
          d,
          0,
          0,
          N,
          dst_group_d,
          moduliQ);
    }

    if (d + 1 < composite_degree) {
      const_mult_batch_(
          temp_ptr,
          qj,
          qj_psinv,
          d + 1,
          composite_degree - (d + 1),
          d + 1,
          d + 1,
          N,
          dst_group_d,
          moduliQ);
    }

    Tensor row_mod = qjProductD_mod.select(0, d).contiguous();       // [L0]
    Tensor row_psinv = qjProductD_psinv.select(0, d).contiguous();   // [L0]

    uint64_t* src_group_init =
        temp_ptr + (static_cast<int64_t>(init_element_index) * L0 * N);

    if (composite_degree < L0) {
      const_mult_batch_(
          src_group_init,
          row_mod,
          row_psinv,
          composite_degree,
          L0 - composite_degree,
          composite_degree,
          composite_degree,
          N,
          dst_group_d,
          moduliQ);
    }

    const_mult_batch_(
        src_group_init,
        row_mod,
        row_psinv,
        d,
        1,
        d,
        d,
        N,
        dst_group_d,
        moduliQ);

    auto moduliQ_ptr = reinterpret_cast<uint64_t*>(moduliQ.data_ptr<uint64_t>());
    for (int64_t l = 0; l < L0; l++) {
      for (int64_t n = 0; n < N; n++) {
        uint64_t* input0 = temp_ptr + (d * L0 + l) * N + n;
        uint64_t* input1 = temp_ptr + l * N + n;
        uint64_t* out = res_ptr + l * N + n;
        *out = (*input0 + *input1) % moduliQ_ptr[l];
      }
    }
  }

  NTT_impl(
      res_ptr,
      0,
      L0,
      N,
      param_power_of_roots_shoup,
      moduliQ,
      param_power_of_roots);
}

Tensor extend_ciphertext_cpu(
    const Tensor& res,
    const Tensor& in,
    int64_t N,
    int64_t L0,
    int64_t level,
    int64_t composite_degree,
    int64_t qjProduct,
    const Tensor& qj,
    const Tensor& qhat_inv_modqj,
    const Tensor& qjProductD,
    const Tensor& qjProduct_mod, 
    const Tensor& qjProduct_psinv,
    const Tensor& qj_psinv,
    const Tensor& qhat_inv_modqj_mod,
    const Tensor& qhat_inv_modqj_psinv,
    const Tensor& qjProductD_psinv,
    const Tensor& moduliQ,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_power_of_roots,
    const Tensor& barret_ratio,
    const Tensor& barret_k) {
  TORCH_INTERNAL_ASSERT(in.dim() == 4);
  TORCH_INTERNAL_ASSERT(in.sizes()[3] == N);
  TORCH_INTERNAL_ASSERT(in.sizes()[2] >= composite_degree);

  const auto num_cv = in.sizes()[0];
  const auto num_cipher = in.sizes()[1];
  Tensor out = at::empty({num_cv, num_cipher, L0, N}, in.options());

  for (int64_t cv_id = 0; cv_id < num_cv; ++cv_id) {
    for (int64_t cipher_id = 0; cipher_id < num_cipher; ++cipher_id) {
      Tensor in_slice = in.select(0, cv_id)
                            .select(0, cipher_id)
                            .narrow(0, 0, composite_degree)
                            .contiguous()
                            .reshape({composite_degree * N});
      Tensor out_slice =
          out.select(0, cv_id).select(0, cipher_id).reshape({L0 * N});
      Tensor temp = at::zeros(
          {(composite_degree + 1) * L0 * N}, in.options().dtype(kUInt64));
      Tensor ctxtDCRT_modq =
          at::zeros({composite_degree * L0 * N}, in.options().dtype(kUInt64));

      extend_ciphertext_template_precomp(
          out_slice,
          in_slice,
          temp,
          ctxtDCRT_modq,
          N,
          L0,
          level,
          composite_degree,
          qj,
          qj_psinv,
          qhat_inv_modqj_mod,
          qhat_inv_modqj_psinv,
          qjProduct_mod,
          qjProduct_psinv,
          qjProductD,
          qjProductD_psinv,
          moduliQ,
          inverse_power_of_roots_div_two,
          inverse_scaled_power_of_roots_div_two,
          param_power_of_roots_shoup,
          param_power_of_roots);
    }
  }

  return out;
}

} // namespace at::native
