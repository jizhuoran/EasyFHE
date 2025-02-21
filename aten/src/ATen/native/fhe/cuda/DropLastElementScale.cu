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

#include "ATen/native/fhe/cuda/arithmetic.h"
#include "ATen/native/fhe/cuda/CommonOperation.h"

#pragma clang diagnostic ignored "-Wmissing-prototypes"

namespace at::native {

static void drop_last_element_scale_template(
    const Tensor& from,
    Tensor& workspace,
    int64_t curr_limbs,
    int64_t l,
    int64_t L,
    int64_t N,
    const Tensor& param_primes,
    const Tensor& param_barret_ratio,
    const Tensor& param_barret_k,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_power_of_roots,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& qlql_inv_mod_ql_div_ql_mod_q,
    const Tensor& qlql_inv_mod_ql_div_ql_mod_q_shoup,
    const Tensor& q_inv_mod_q,
    const Tensor& q_inv_mod_q_shoup,
    Tensor& res) {
  const int end_length = curr_limbs - 1;
  auto from_ptr = reinterpret_cast<uint64_t*>(from.data_ptr<uint64_t>());
  auto workspace_ptr = reinterpret_cast<uint64_t*>(workspace.data_ptr<uint64_t>());
  auto to_ptr = reinterpret_cast<uint64_t*>(res.data_ptr<uint64_t>());
  
  iNTT_impl(
      from_ptr,
      workspace_ptr,
      end_length,
      1,
      curr_limbs,
      L,
      N,
      inverse_power_of_roots_div_two,
      param_primes,
      inverse_scaled_power_of_roots_div_two);

  switch_modulus(
      to_ptr,
      workspace_ptr + N * end_length,
      param_primes,
      param_barret_ratio,
      param_barret_k,
      curr_limbs - 1,
      curr_limbs - 1,
      N);

  int start_op2_idx = (L - curr_limbs + l) * (L - 1);
  const_mult_batch(
      to_ptr,
      to_ptr,
      qlql_inv_mod_ql_div_ql_mod_q.data_ptr<uint64_t>() + start_op2_idx,
      qlql_inv_mod_ql_div_ql_mod_q_shoup.data_ptr<uint64_t>() + start_op2_idx,
      param_primes.data_ptr<uint64_t>(),
      curr_limbs - 1,
      N);

  NTT_impl(
      to_ptr,
      to_ptr,
      end_length,
      N,
      param_power_of_roots_shoup.data_ptr<uint64_t>(),
      param_primes.data_ptr<uint64_t>(),
      param_power_of_roots.data_ptr<uint64_t>());

  start_op2_idx = (curr_limbs - 1) * (L);
  const_mult_batch(
      workspace_ptr,
      from_ptr,
      q_inv_mod_q.data_ptr<uint64_t>() + start_op2_idx,
      q_inv_mod_q_shoup.data_ptr<uint64_t>() + start_op2_idx,
      param_primes.data_ptr<uint64_t>(),
      end_length,
      N);


  vadd_mod(
  N,
  end_length,
  to_ptr,
  to_ptr,
  workspace_ptr,
  param_primes.data_ptr<uint64_t>());


  // vec_add_mod_batch(
  //     to_ptr,
  //     to_ptr,
  //     workspace_ptr,
  //     param_primes,
  //     param_barret_ratio,
  //     param_barret_k,
  //     end_length,
  //     N);
}

Tensor drop_last_element_scale_cuda(
    const Tensor& to,
    const Tensor& from,
    int64_t curr_limbs,
    int64_t l,
    int64_t L,
    int64_t N,
    const Tensor& param_primes,
    const Tensor& param_barret_ratio,
    const Tensor& param_barret_k,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_power_of_roots,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& qlql_inv_mod_ql_div_ql_mod_q,
    const Tensor& qlql_inv_mod_ql_div_ql_mod_q_shoup,
    const Tensor& q_inv_mod_q,
    const Tensor& q_inv_mod_q_shoup) {
  auto res = at::empty((curr_limbs - 1) * N, to.options());
  auto workspace = at::empty(curr_limbs * N, to.options());

  drop_last_element_scale_template(
      from,
      workspace,
      curr_limbs,
      l,
      L,
      N,
      param_primes,
      param_barret_ratio,
      param_barret_k,
      param_power_of_roots_shoup,
      param_power_of_roots,
      inverse_power_of_roots_div_two,
      inverse_scaled_power_of_roots_div_two,
      qlql_inv_mod_ql_div_ql_mod_q,
      qlql_inv_mod_ql_div_ql_mod_q_shoup,
      q_inv_mod_q,
      q_inv_mod_q_shoup,
      res);

  return res;
}

Tensor& drop_last_element_scale_cuda_(
    Tensor& to,
    const Tensor& from,
    int64_t curr_limbs,
    int64_t l,
    int64_t L,
    int64_t N,
    const Tensor& param_primes,
    const Tensor& param_barret_ratio,
    const Tensor& param_barret_k,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_power_of_roots,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& qlql_inv_mod_ql_div_ql_mod_q,
    const Tensor& qlql_inv_mod_ql_div_ql_mod_q_shoup,
    const Tensor& q_inv_mod_q,
    const Tensor& q_inv_mod_q_shoup) {
  to.resize_({(curr_limbs - 1) * N});

//   drop_last_element_scale_template(
//       from,
//       curr_limbs,
//       l,
//       L,
//       N,
//       param_primes,
//       param_barret_ratio,
//       param_barret_k,
//       param_power_of_roots_shoup,
//       param_power_of_roots,
//       inverse_power_of_roots_div_two,
//       inverse_scaled_power_of_roots_div_two,
//       qlql_inv_mod_ql_div_ql_mod_q,
//       qlql_inv_mod_ql_div_ql_mod_q_shoup,
//       q_inv_mod_q,
//       q_inv_mod_q_shoup,
//       to);

  return to;
}

Tensor& drop_last_element_scale_cuda_out(
    const Tensor& to,
    const Tensor& from,
    int64_t curr_limbs,
    int64_t l,
    int64_t L,
    int64_t N,
    const Tensor& param_primes,
    const Tensor& param_barret_ratio,
    const Tensor& param_barret_k,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_power_of_roots,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& qlql_inv_mod_ql_div_ql_mod_q,
    const Tensor& qlql_inv_mod_ql_div_ql_mod_q_shoup,
    const Tensor& q_inv_mod_q,
    const Tensor& q_inv_mod_q_shoup,
    Tensor& res) {
  res.resize_({(curr_limbs - 1) * N});

//   drop_last_element_scale_template(
//       from,
//       curr_limbs,
//       l,
//       L,
//       N,
//       param_primes,
//       param_barret_ratio,
//       param_barret_k,
//       param_power_of_roots_shoup,
//       param_power_of_roots,
//       inverse_power_of_roots_div_two,
//       inverse_scaled_power_of_roots_div_two,
//       qlql_inv_mod_ql_div_ql_mod_q,
//       qlql_inv_mod_ql_div_ql_mod_q_shoup,
//       q_inv_mod_q,
//       q_inv_mod_q_shoup,
//       res);

  return res;
}

} // namespace at::native