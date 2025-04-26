#include "ATen/native/fhe/cpu/KeySwitch.h"
#include <ATen/Dispatch_v2.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/interned_strings.h>
#include <ATen/ops/copy.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>
#include <immintrin.h>
#include <omp.h>
#include <iostream>
#include "ATen/native/fhe/cpu/NttImpl.h"
#pragma clang diagnostic ignored "-Wmissing-prototypes"

namespace fhe {

void const_mult_batch(
    size_t degree,
    const uint64_t* primes,
    uint64_t* op1,
    const uint64_t* op2,
    const uint64_t* op2_psinv,
    const int start_prime_idx,
    const int batch,
    const int start_op1_idx,
    const int start_op2_idx,
    uint64_t* to) {
  const int max_threads = omp_get_max_threads();
  omp_set_num_threads(max_threads);
#pragma omp parallel for num_threads(max_threads)
  for (int i = 0; i < degree * batch; i++) {
    const int op2_idx = start_op2_idx + i / degree;
    const int prime_idx = i / degree + start_prime_idx;
    const auto prime = primes[prime_idx];

    uint64_t out = mul_and_reduce_shoup(
        op1[start_op1_idx * degree + i],
        op2[op2_idx],
        op2_psinv[op2_idx],
        prime);

    if (out >= prime)
      out -= prime;
    to[start_op1_idx * degree + i] = out;
  }
}

uint128_t4 accumulate_in_modup(
    const uint64_t* ptr,
    const int degree,
    const uint64_t* hat_mod_end,
    const int start_length, // sizeP
    const int degree_idx,
    const int hat_mod_end_idx) {
  uint128_t4 accum{0};

  for (int i = 0; i < start_length; i++) {
    const uint64_t op2 = hat_mod_end[hat_mod_end_idx * start_length + i];
    uint128_t4 out;
    uint64_t op1_x, op1_y, op1_z, op1_w;

    op1_x = ptr[i * degree + degree_idx];
    op1_y = ptr[i * degree + degree_idx + 1];
    op1_z = ptr[i * degree + degree_idx + 2];
    op1_w = ptr[i * degree + degree_idx + 3];

    out.x = mult_64_64_128(op1_x, op2);
    inplace_add_128_128(out.x, accum.x);
    out.y = mult_64_64_128(op1_y, op2);
    inplace_add_128_128(out.y, accum.y);
    out.z = mult_64_64_128(op1_z, op2);
    inplace_add_128_128(out.z, accum.z);
    out.w = mult_64_64_128(op1_w, op2);
    inplace_add_128_128(out.w, accum.w);
  }
  return accum;
}

void modup_step_two_kernel(
    const uint64_t* ptr,
    const int begin_idx,
    const int degree, // ringDim
    const int alpha,
    const int curr_limbs,
    const int level,
    const uint64_t* primes,
    const uint64_t* barrett_ratios,
    const uint64_t* barrett_Ks,
    const uint64_t* hat_mod_end,
    const int hat_mod_end_size,
    const uint64_t start_length, // sizeP
    const uint64_t end_length, // sizeQ
    uint64_t* to) {
  constexpr const int unroll_number = 4;
  std::vector<uint64_t> s_hat_mod_end_vec(hat_mod_end_size);
  uint64_t* s_hat_mod_end = s_hat_mod_end_vec.data();
  const int max_threads = omp_get_max_threads();
  omp_set_num_threads(max_threads);
  for (int i = 0; i < hat_mod_end_size; ++i) {
    s_hat_mod_end[i] = hat_mod_end[i];
  }
#pragma omp parallel for num_threads(max_threads)
  for (int i = 0; i < (degree * end_length + unroll_number - 1) / unroll_number;
       i++) {
    const int degree_idx = unroll_number * (i / end_length);
    const int hat_mod_end_idx = i % end_length; // j
    const int out_idx =
        hat_mod_end_idx + ((hat_mod_end_idx >= begin_idx) ? start_length : 0);
    uint128_t4 accum = accumulate_in_modup(
        ptr, degree, s_hat_mod_end, alpha, degree_idx, hat_mod_end_idx);
    int gap = level - curr_limbs;
    int prime_idx = out_idx +
        (((out_idx >= 0 && out_idx < begin_idx) ||
          (out_idx >= (begin_idx + start_length) && out_idx < curr_limbs))
             ? 0
             : gap);
    const auto prime = primes[prime_idx];
    const auto barret_ratio = barrett_ratios[prime_idx];
    const auto barret_k = barrett_Ks[prime_idx];
    {
      // First store operation
      uint64_t out1 =
          barret_reduction_128_64(accum.x, prime, barret_ratio, barret_k);
      uint64_t out2 =
          barret_reduction_128_64(accum.y, prime, barret_ratio, barret_k);

      to[out_idx * degree + degree_idx] = out1;
      to[out_idx * degree + degree_idx + 1] = out2;
    }
    {
      uint64_t out3 =
          barret_reduction_128_64(accum.z, prime, barret_ratio, barret_k);
      uint64_t out4 =
          barret_reduction_128_64(accum.w, prime, barret_ratio, barret_k);
      to[out_idx * degree + degree_idx + 2] = out3;
      to[out_idx * degree + degree_idx + 3] = out4;
    }
  }
}

void moddown_kernel(
    int degree_,
    uint64_t* d_primes,
    uint64_t* d_barret_ratio,
    uint64_t* d_barret_k,
    int log_degree_,
    const uint64_t* ptr,
    const uint64_t* hat_mod_end,
    const int hat_mod_end_size,
    const uint64_t start_length,
    const uint64_t end_length,
    uint64_t* to) {
  constexpr const int unroll_number = 4;
  std::vector<uint64_t> s_hat_mod_end_vec(hat_mod_end_size);
  uint64_t* s_hat_mod_end = s_hat_mod_end_vec.data();
  const int max_threads = omp_get_max_threads();
  omp_set_num_threads(max_threads);
  for (int i = 0; i < hat_mod_end_size; ++i) {
    s_hat_mod_end[i] = hat_mod_end[i];
  }
#pragma omp parallel for schedule(static) num_threads(max_threads)
  for (int i = 0;
       i < (degree_ * end_length + unroll_number - 1) / unroll_number;
       i++) {
    const int degree_idx = unroll_number * (i / end_length);
    const int out_prime_idx = i % end_length;
    uint128_t4 accum = accumulate_in_modup(
        ptr, degree_, s_hat_mod_end, start_length, degree_idx, out_prime_idx);
    const auto prime = d_primes[out_prime_idx];
    const auto barret_ratio = d_barret_ratio[out_prime_idx];
    const auto barret_k = d_barret_k[out_prime_idx];
    {
      uint64_t out =
          barret_reduction_128_64(accum.x, prime, barret_ratio, barret_k);
      uint64_t out2 =
          barret_reduction_128_64(accum.y, prime, barret_ratio, barret_k);
      to[out_prime_idx * degree_ + degree_idx] = out;
      to[out_prime_idx * degree_ + degree_idx + 1] = out2;
    }
    {
      uint64_t out3 =
          barret_reduction_128_64(accum.z, prime, barret_ratio, barret_k);
      uint64_t out4 =
          barret_reduction_128_64(accum.w, prime, barret_ratio, barret_k);
      to[out_prime_idx * degree_ + degree_idx + 2] = out3;
      to[out_prime_idx * degree_ + degree_idx + 3] = out4;
    }
  }
}

void negateInplace_(
    size_t degree,
    size_t log_degree,
    size_t batch,
    const uint64_t* primes,
    uint64_t* op) {
  for (int i = 0; i < batch * degree; i++) {
    const int prime_idx = i >> log_degree;
    const uint64_t prime = primes[prime_idx];
    if (op[i] != 0)
      op[i] = prime - op[i];
  }
}

void subInplace_(
    size_t degree,
    size_t batch,
    const uint64_t* primes,
    uint64_t* op1,
    const uint64_t* op2) {
  for (int i = 0; i < batch * degree; i++) {
    const int prime_idx = i / degree;
    const uint64_t prime = primes[prime_idx];
    if (op1[i] >= op2[i]) {
      op1[i] -= op2[i];
    } else {
      op1[i] = prime - (op2[i] - op1[i]);
    }
  }
}

void vec_add_mod_batch_(
    int degree_,
    uint64_t* d_primes,
    uint64_t* d_barret_ratio,
    uint64_t* d_barret_k,
    const uint64_t* op1,
    const uint64_t* op2,
    const uint64_t batch,
    uint64_t* to) {
  const int max_threads = omp_get_max_threads();
  omp_set_num_threads(max_threads);
#pragma omp parallel for schedule(static) num_threads(max_threads)
  for (int i = 0; i < batch * degree_; i++) {
    // const int degree_idx = unroll_number * (i / end_length);
    const int out_prime_idx = i / degree_;
    // const int in_idx = i % degree_;
    const auto prime = d_primes[out_prime_idx];
    const auto barret_ratio = d_barret_ratio[out_prime_idx];
    const auto barret_k = d_barret_k[out_prime_idx];
    barret_reduction_64_64(
        op1[i] + op2[i], to[i], prime, barret_ratio, barret_k);
  }
}

void vec_mod_batch_(
    int degree_,
    uint64_t* d_primes,
    uint64_t* d_barret_ratio,
    uint64_t* d_barret_k,
    const uint64_t* op1,
    const uint64_t batch,
    uint64_t* to) {
  STRIDED_LOOP_START((degree_ * batch), i);
  const int out_prime_idx = i / degree_;
  const int op1_idx = i % degree_;
  const auto prime = d_primes[out_prime_idx];
  const auto barret_ratio = d_barret_ratio[out_prime_idx];
  const auto barret_k = d_barret_k[out_prime_idx];
  barret_reduction_64_64(op1[op1_idx], to[i], prime, barret_ratio, barret_k);

  STRIDED_LOOP_END;
}

void switch_modulus_(
    size_t degree,
    size_t batch,
    const size_t old_prime_idx,
    const uint64_t* primes,
    const uint64_t* ptr,
    uint64_t* to) {
  const auto old_modulus = primes[old_prime_idx];
  const auto old_modulus_by_two = old_modulus >> 1;
  const int max_threads = omp_get_max_threads();
  omp_set_num_threads(max_threads);
  const int total = batch * degree;
#pragma omp parallel for schedule(static) num_threads(max_threads)
  for (int i = 0; i < total; ++i) {
    const auto new_modulus_idx = i / degree;
    const auto nm = primes[new_modulus_idx];

    // 计算 diff 的优化版本 (避免分支)
    const auto modulus_diff =
        (old_modulus > nm) ? (nm - (old_modulus % nm)) : (nm - old_modulus);

    const int input_idx = i % degree;
    const uint64_t tmp =
        (ptr[input_idx] > old_modulus_by_two) ? modulus_diff : 0;

    // 计算结果并处理模数
    uint64_t val = ptr[input_idx] + tmp;
    if (nm <= old_modulus) {
      val %= nm; // 当 nm <= old_modulus 时直接取模
    }
    to[i] = val;
  }
}

} // namespace fhe

namespace at::native {

Tensor iNTT_cpu(
    const Tensor& op,
    int64_t start_prime_idx,
    int64_t batch,
    int64_t param_degree,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& param_primes,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    int64_t curr_limbs,
    int64_t level) {
  auto res = op.clone();
  auto op_ptr = reinterpret_cast<uint64_t*>(res.data_ptr<uint64_t>());
  iNTT_impl(
      op_ptr,
      op_ptr,
      start_prime_idx,
      batch,
      curr_limbs,
      level,
      param_degree,
      inverse_power_of_roots_div_two,
      param_primes,
      inverse_scaled_power_of_roots_div_two);

  return res;
}

Tensor& iNTT_cpu_(
    Tensor& op,
    int64_t start_prime_idx,
    int64_t batch,
    int64_t param_degree,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& param_primes,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    int64_t curr_limbs,
    int64_t level) {
  auto op_ptr = reinterpret_cast<uint64_t*>(op.data_ptr<uint64_t>());
  iNTT_impl(
      op_ptr,
      op_ptr,
      start_prime_idx,
      batch,
      curr_limbs,
      level,
      param_degree,
      inverse_power_of_roots_div_two,
      param_primes,
      inverse_scaled_power_of_roots_div_two);

  return op;
}

Tensor& iNTT_cpu_out(
    const Tensor& op,
    int64_t start_prime_idx,
    int64_t batch,
    int64_t param_degree,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& param_primes,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    int64_t curr_limbs,
    int64_t level,
    Tensor& res) {
  auto op_ptr = reinterpret_cast<uint64_t*>(res.data_ptr<uint64_t>());
  iNTT_impl(
      op_ptr,
      op_ptr,
      start_prime_idx,
      batch,
      curr_limbs,
      level,
      param_degree,
      inverse_power_of_roots_div_two,
      param_primes,
      inverse_scaled_power_of_roots_div_two);

  return res;
}

Tensor NTT_cpu(
    const Tensor& op,
    int64_t start_prime_idx,
    int64_t batch,
    int64_t param_degree,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_primes,
    const Tensor& param_power_of_roots) {
  auto res = op.clone();
  auto op_ptr = reinterpret_cast<uint64_t*>(res.data_ptr<uint64_t>());
  NTT_impl(
      op_ptr,
      op_ptr,
      start_prime_idx,
      batch,
      param_degree,
      param_power_of_roots_shoup,
      param_primes,
      param_power_of_roots);

  return res;
}

Tensor& NTT_cpu_(
    Tensor& op,
    int64_t start_prime_idx,
    int64_t batch,
    int64_t param_degree,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_primes,
    const Tensor& param_power_of_roots) {
  auto op_ptr = reinterpret_cast<uint64_t*>(op.data_ptr<uint64_t>());
  NTT_impl(
      op_ptr,
      op_ptr,
      start_prime_idx,
      batch,
      param_degree,
      param_power_of_roots_shoup,
      param_primes,
      param_power_of_roots);

  return op;
}

Tensor& NTT_cpu_out(
    const Tensor& op,
    int64_t start_prime_idx,
    int64_t batch,
    int64_t param_degree,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_primes,
    const Tensor& param_power_of_roots,
    Tensor& res) {
  auto op_ptr = reinterpret_cast<uint64_t*>(res.data_ptr<uint64_t>());
  NTT_impl(
      op_ptr,
      op_ptr,
      start_prime_idx,
      batch,
      param_degree,
      param_power_of_roots_shoup,
      param_primes,
      param_power_of_roots);

  return res;
}

int myGetMSB(int64_t x) {
  if (x == 0)
    return -1; // No set bit, return -1

  int position = 0;
  while (x > 0) {
    x >>= 1; // Shift right by 1 bit
    position++; // Increment the position
  }
  return position; // The MSB is 1 less than the number of shifts
}
static void NTT_except_some_range_impl(
    uint64_t* op_ptr,
    int64_t start_prime_idx,
    int64_t batch,
    int64_t param_degree,
    int64_t excluded_range_start,
    int64_t excluded_range_size,
    int64_t curr_limbs,
    int64_t level,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_primes,
    const Tensor& param_power_of_roots) {
  auto excluded_range_end = excluded_range_start + excluded_range_size;
  const int max_threads = omp_get_max_threads();
  omp_set_num_threads(max_threads);

  auto param_power_of_roots_shoup_ptr = reinterpret_cast<uint64_t*>(
      param_power_of_roots_shoup.data_ptr<uint64_t>());
  auto param_primes_ptr =
      reinterpret_cast<uint64_t*>(param_primes.data_ptr<uint64_t>());
  auto param_power_of_roots_ptr =
      reinterpret_cast<uint64_t*>(param_power_of_roots.data_ptr<uint64_t>());
  int gap = level - curr_limbs;
  const int64_t n = param_degree >> 1;
#pragma omp parallel for schedule(static) num_threads(max_threads)
  for (int bach = 0; bach < batch; ++bach) {
    uint64_t primeidx = batch - 1 - bach;
    +start_prime_idx;
    if (primeidx >= excluded_range_start && primeidx < excluded_range_end)
      continue;
    uint64_t prime_idx =
        primeidx + ((primeidx >= 0 && primeidx < curr_limbs) ? 0 : gap);
    uint64_t modulus = param_primes_ptr[prime_idx];
    uint64_t base_prime_idx = prime_idx * param_degree;
    uint64_t base = primeidx * param_degree;
    for (uint32_t m = 1, t = n, logt = myGetMSB(t); m < n;
         m <<= 1, t >>= 1, --logt) {
      for (uint32_t i = 0; i < m; ++i) {
        auto omega = param_power_of_roots_ptr[i + m + base_prime_idx]; // S
        auto preconOmega = param_power_of_roots_shoup_ptr
            [i + m + base_prime_idx]; // NEEDED IN COMPUTE F[j+t]*S MOD Q
        for (uint32_t j1 = (i << logt), j2 = j1 + t; j1 < j2; ++j1) {
          uint64_t a1 = (op_ptr)[j1 + 0 + base];
          uint64_t b1 = (op_ptr)[j1 + t + base];
          fhe::butt_ntt_local(a1, b1, omega, preconOmega, modulus);
          (op_ptr)[j1 + 0 + base] = a1;
          (op_ptr)[j1 + t + base] = b1;
        }
      }
    }
    for (uint32_t i = 0; i < (n << 1); i += 2) {
      auto omega = param_power_of_roots_ptr[(i >> 1) + n + base_prime_idx];
      auto preconOmega =
          param_power_of_roots_shoup_ptr[(i >> 1) + n + base_prime_idx];
      uint64_t a1 = (op_ptr)[i + 0 + base];
      uint64_t b1 = (op_ptr)[i + 1 + base];
      fhe::butt_ntt_local(a1, b1, omega, preconOmega, modulus);
      for (int a = 0; a < 3; a++) {
        if (b1 > modulus) {
          b1 -= modulus;
        }
        if (a1 > modulus) {
          a1 -= modulus;
        }
      }
      (op_ptr)[i + 0 + base] = a1;
      (op_ptr)[i + 1 + base] = b1;
    }
  }
}
void const_mult_batch_(
    uint64_t* op1_ptr,
    const Tensor& op2,
    const Tensor& op2_psinv,
    int64_t start_prime_idx,
    int64_t batch,
    int64_t start_op1_idx,
    int64_t start_op2_idx,
    int64_t param_degree,
    uint64_t* res_ptr,
    const Tensor& primes) {
  AT_DISPATCH_V2(
      op2.scalar_type(),
      "const_mult_batch_",
      AT_WRAP([&]() {
        auto op2_ptr = reinterpret_cast<uint64_t*>(op2.data_ptr<uint64_t>());
        auto op2_psinv_ptr =
            reinterpret_cast<uint64_t*>(op2_psinv.data_ptr<uint64_t>());
        auto primes_ptr =
            reinterpret_cast<uint64_t*>(primes.data_ptr<uint64_t>());
        const int block_dim = 256;
        const int grid_dim = param_degree * batch / block_dim;
        fhe::const_mult_batch(
            (int)param_degree,
            primes_ptr,
            op1_ptr,
            op2_ptr,
            op2_psinv_ptr,
            (int)start_prime_idx,
            (int)batch,
            (int)start_op1_idx,
            (int)start_op2_idx,
            res_ptr);
      }),
      kUInt64);
}

static void modup_matmul_(
    uint64_t* ptr,
    int64_t beta_idx,
    uint64_t* to_ptr,
    const Tensor& primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    const int64_t param_alpha_,
    const int64_t param_degree_,
    const Tensor& prod_q_i_mod_q_j__,
    int64_t curr_limbs,
    int64_t level_) {
  const int unroll_factor = 4;
  const int begin_idx = (int)beta_idx * (int)param_alpha_;
  int start_length = ((begin_idx + param_alpha_) > curr_limbs)
      ? (curr_limbs - begin_idx)
      : param_alpha_;
  const int end_length = curr_limbs + param_alpha_ - start_length;
  int grid_dim{(int)param_degree_ * end_length / 256 / unroll_factor};
  int block_dim{256};
  const auto& prod_q_i_mod_q_j = prod_q_i_mod_q_j__[beta_idx];
  AT_DISPATCH_V2(
      kUInt64,
      "modup_matmul_",
      AT_WRAP([&]() {
        auto primes_ptr =
            reinterpret_cast<uint64_t*>(primes.data_ptr<uint64_t>());
        auto barret_ratio_ptr =
            reinterpret_cast<uint64_t*>(barret_ratio.data_ptr<uint64_t>());
        auto barret_k_ptr =
            reinterpret_cast<uint64_t*>(barret_k.data_ptr<uint64_t>());
        auto prod_q_i_mod_q_j_ptr =
            reinterpret_cast<uint64_t*>(prod_q_i_mod_q_j.data_ptr<uint64_t>());
        fhe::modup_step_two_kernel(
            ptr,
            begin_idx,
            param_degree_,
            param_alpha_,
            curr_limbs,
            level_,
            primes_ptr,
            barret_ratio_ptr,
            barret_k_ptr,
            prod_q_i_mod_q_j_ptr,
            prod_q_i_mod_q_j.size(-1),
            start_length,
            end_length,
            to_ptr);
      }),
      kUInt64);
}

static void modup_impl_(
    uint64_t* from_ptr,
    uint64_t* to_ptr,
    int idx,
    int curr_limbs,
    int level,
    const Tensor& hat_inverse_vec__,
    const Tensor& hat_inverse_vec_shoup__,
    const int64_t param_degree_,
    const int64_t param_alpha_,
    const Tensor& param_primes__,
    const Tensor& param_barret_ratio__,
    const Tensor& param_barret_k__,
    const Tensor& prod_q_i_mod_q_j__,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_power_of_roots) {
  int num_moduli_after_modup = curr_limbs + param_alpha_;
  size_t begin_idx = idx * param_alpha_;
  size_t in_C_L_len = ((begin_idx + param_alpha_) > curr_limbs)
      ? (curr_limbs - begin_idx)
      : param_alpha_;

  auto hat_inverse_vec =
      hat_inverse_vec__[idx * param_alpha_ + (in_C_L_len - 1)];
  auto hat_inverse_vec_psinv =
      hat_inverse_vec_shoup__[idx * param_alpha_ + (in_C_L_len - 1)];
  const int max_threads = omp_get_max_threads();
  omp_set_num_threads(max_threads);

  memcpy(
      to_ptr + (param_degree_ * begin_idx),
      from_ptr,
      8 * in_C_L_len * param_degree_);

  iNTT_impl(
      to_ptr,
      to_ptr,
      begin_idx,
      in_C_L_len,
      curr_limbs,
      level,
      param_degree_,
      inverse_power_of_roots_div_two,
      param_primes__,
      inverse_scaled_power_of_roots_div_two);

  // const_mult_batch_(
  //     to_ptr,
  //     hat_inverse_vec,
  //     hat_inverse_vec_psinv,
  //     begin_idx,
  //     in_C_L_len,
  //     begin_idx,
  //     0,
  //     param_degree_,
  //     to_ptr,
  //     param_primes__);
  auto op2_ptr =
      reinterpret_cast<uint64_t*>(hat_inverse_vec.data_ptr<uint64_t>());
  auto op2_psinv_ptr =
      reinterpret_cast<uint64_t*>(hat_inverse_vec_psinv.data_ptr<uint64_t>());
  auto primes_ptr =
      reinterpret_cast<uint64_t*>(param_primes__.data_ptr<uint64_t>());
#pragma omp parallel for num_threads(max_threads)
  for (int i = 0; i < param_degree_ * in_C_L_len; i++) {
    const int op2_idx = 0 + i / param_degree_;
    const int prime_idx = i / param_degree_ + begin_idx;
    const auto prime = primes_ptr[prime_idx];
    uint64_t out = fhe::mul_and_reduce_shoup(
        to_ptr[begin_idx * param_degree_ + i],
        op2_ptr[op2_idx],
        op2_psinv_ptr[op2_idx],
        prime);
    if (out >= prime)
      out -= prime;
    to_ptr[begin_idx * param_degree_ + i] = out;
  }
  modup_matmul_(
      to_ptr + param_degree_ * begin_idx,
      idx,
      to_ptr,
      param_primes__,
      param_barret_ratio__,
      param_barret_k__,
      param_alpha_,
      param_degree_,
      prod_q_i_mod_q_j__,
      curr_limbs,
      level);

  NTT_except_some_range_impl(
      to_ptr,
      0,
      num_moduli_after_modup,
      param_degree_,
      begin_idx,
      in_C_L_len,
      curr_limbs,
      level,
      param_power_of_roots_shoup,
      param_primes__,
      param_power_of_roots);

  memcpy(
      to_ptr + (param_degree_ * begin_idx),
      from_ptr,
      8 * in_C_L_len * param_degree_);
}

static void modup(
    uint64_t* in_ptr,
    int64_t curr_limbs,
    int64_t level,
    const Tensor& hat_inverse_vec__,
    const Tensor& hat_inverse_vec_shoup__,
    const Tensor& prod_q_i_mod_q_j__,
    const Tensor& param_primes__,
    const Tensor& param_barret_ratio__,
    const Tensor& param_barret_k__,
    int64_t beta,
    int64_t param_degree_,
    int64_t param_alpha_,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_power_of_roots,
    uint64_t* out_ptr) {
  int num_moduli_after_modup = curr_limbs + param_alpha_;
  for (int i = 0; i < beta; ++i) {
    modup_impl_(
        in_ptr + (param_alpha_ * param_degree_ * i),
        out_ptr + (num_moduli_after_modup * param_degree_) * i,
        i,
        curr_limbs,
        level,
        hat_inverse_vec__,
        hat_inverse_vec_shoup__,
        param_degree_,
        param_alpha_,
        param_primes__,
        param_barret_ratio__,
        param_barret_k__,
        prod_q_i_mod_q_j__,
        inverse_power_of_roots_div_two,
        inverse_scaled_power_of_roots_div_two,
        param_power_of_roots_shoup,
        param_power_of_roots);
  }
}

Tensor modup_cpu(
    const Tensor& in,
    int64_t curr_limbs,
    int64_t L,
    const Tensor& hat_inverse_vecs,
    const Tensor& hat_inverse_vec_shoups,
    const Tensor& prod_q_i_mod_q_js,
    const Tensor& primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    int64_t beta,
    int64_t N,
    int64_t alpha,
    const Tensor& power_of_roots_shoup,
    const Tensor& power_of_roots,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two) {
  int64_t sizeQP = primes.numel();
  int64_t sizeP = sizeQP - L;
  auto out = at::empty(beta * (curr_limbs + sizeP) * N, in.options());
  auto in_ptr = reinterpret_cast<uint64_t*>(in.data_ptr<uint64_t>());
  auto out_ptr = reinterpret_cast<uint64_t*>(out.data_ptr<uint64_t>());
  modup(
      in_ptr,
      curr_limbs,
      L,
      hat_inverse_vecs,
      hat_inverse_vec_shoups,
      prod_q_i_mod_q_js,
      primes,
      barret_ratio,
      barret_k,
      beta,
      N,
      alpha,
      inverse_power_of_roots_div_two,
      inverse_scaled_power_of_roots_div_two,
      power_of_roots_shoup,
      power_of_roots,
      out_ptr);
  return out;
}

static void NegateInplace(
    uint64_t* op1,
    const int batch,
    const Tensor& primes,
    const int64_t param_degree,
    const int64_t param_log_degree) {
  AT_DISPATCH_V2(
      kUInt64,
      "NegateInplace",
      AT_WRAP([&]() {
        const int block_dim = 256;
        const int grid_dim = param_degree * batch / block_dim;
        auto primes_ptr =
            reinterpret_cast<uint64_t*>(primes.data_ptr<uint64_t>());
        fhe::negateInplace_(
            param_degree, param_log_degree, batch, primes_ptr, op1);
      }),
      kUInt64);
}

void SubInplace(
    uint64_t* op1,
    const uint64_t* op2,
    const int64_t batch,
    const int64_t param_degree,
    const Tensor& primes) {
  AT_DISPATCH_V2(
      kUInt64,
      "SubInplace",
      AT_WRAP([&]() {
        const int block_dim = 256;
        const int grid_dim = param_degree * batch / block_dim;
        auto primes_ptr =
            reinterpret_cast<uint64_t*>(primes.data_ptr<uint64_t>());
        fhe::subInplace_(param_degree, batch, primes_ptr, op1, op2);
      }),
      kUInt64);
}

static void moddown_impl(
    uint64_t* from_ptr,
    const int64_t param_degree,
    const int64_t param_log_degree,
    const int64_t param_alpha_,
    const int64_t start_length,
    const int64_t end_length,
    const Tensor& primes,
    const Tensor& prod_q_i_mod_q_j_moddown,
    const Tensor& param_barret_ratio,
    const Tensor& param_barret_k,
    uint64_t* to_ptr) {
  const auto prod_q_i_mod_q_j = prod_q_i_mod_q_j_moddown[0];
  AT_DISPATCH_V2(
      kUInt64,
      "moddownImpl",
      AT_WRAP([&]() {
        const int block_dim = 256;
        const int grid_dim = param_degree * end_length / block_dim;
        auto ptr = from_ptr + param_degree * end_length;
        auto primes_ptr =
            reinterpret_cast<uint64_t*>(primes.data_ptr<uint64_t>());
        auto param_barret_ratio_ptr = reinterpret_cast<uint64_t*>(
            param_barret_ratio.data_ptr<uint64_t>());
        auto param_barret_k_ptr =
            reinterpret_cast<uint64_t*>(param_barret_k.data_ptr<uint64_t>());
        auto prod_q_i_mod_q_j_ptr =
            reinterpret_cast<uint64_t*>(prod_q_i_mod_q_j.data_ptr<uint64_t>());
        fhe::moddown_kernel(
            param_degree,
            primes_ptr,
            param_barret_ratio_ptr,
            param_barret_k_ptr,
            param_log_degree,
            ptr,
            prod_q_i_mod_q_j_ptr,
            start_length * end_length,
            param_alpha_,
            end_length,
            to_ptr);
      }),
      kUInt64);
}

static void moddown_core_template(
    const Tensor& from,
    int64_t curr_limbs,
    int64_t level,
    int64_t alpha,
    int64_t param_degree,
    int64_t param_log_degree,
    const Tensor& hat_inverse_vec_moddown,
    const Tensor& hat_inverse_vec_shoup_moddown,
    const Tensor& prod_q_i_mod_q_j_moddown,
    const Tensor& prod_inv_moddown,
    const Tensor& prod_inv_shoup_moddown,
    const Tensor& param_primes,
    const Tensor& param_barret_ratio,
    const Tensor& param_barret_k,
    Tensor& res) {
  const int start_length = level + alpha - curr_limbs; // tempK len
  const int end_length = curr_limbs;

  auto hat_inverse_vec = hat_inverse_vec_moddown[0];
  auto hat_inverse_vec_psinv = hat_inverse_vec_shoup_moddown[0];

  auto from_ptr = reinterpret_cast<uint64_t*>(from.data_ptr<uint64_t>());
  auto to_ptr = reinterpret_cast<uint64_t*>(res.data_ptr<uint64_t>());
  const int max_threads = omp_get_max_threads();
  omp_set_num_threads(max_threads);
  // const_mult_batch_(
  //     from_ptr,
  //     hat_inverse_vec,
  //     hat_inverse_vec_psinv,
  //     level,
  //     alpha,
  //     curr_limbs,
  //     0,
  //     param_degree,
  //     from_ptr,
  //     param_primes);
  auto op2_ptr =
      reinterpret_cast<uint64_t*>(hat_inverse_vec.data_ptr<uint64_t>());
  auto op2_psinv_ptr =
      reinterpret_cast<uint64_t*>(hat_inverse_vec_psinv.data_ptr<uint64_t>());
  auto primes_ptr =
      reinterpret_cast<uint64_t*>(param_primes.data_ptr<uint64_t>());
#pragma omp parallel for num_threads(max_threads)
  for (int i = 0; i < param_degree * alpha; i++) {
    const int op2_idx = 0 + i / param_degree;
    const int prime_idx = i / param_degree + level;
    const auto prime = primes_ptr[prime_idx];
    uint64_t out = fhe::mul_and_reduce_shoup(
        from_ptr[curr_limbs * param_degree + i],
        op2_ptr[op2_idx],
        op2_psinv_ptr[op2_idx],
        prime);
    if (out >= prime)
      out -= prime;
    from_ptr[curr_limbs * param_degree + i] = out;
  }

  moddown_impl(
      from_ptr,
      param_degree,
      param_log_degree,
      alpha,
      start_length,
      end_length,
      param_primes,
      prod_q_i_mod_q_j_moddown,
      param_barret_ratio,
      param_barret_k,
      to_ptr);

  const auto& prod_inv = prod_inv_moddown[0];
  const auto& prod_inv_psinv = prod_inv_shoup_moddown[0];

  SubInplace(to_ptr, from_ptr, end_length, param_degree, param_primes);

  NegateInplace(
      to_ptr, end_length, param_primes, param_degree, param_log_degree);

  // const_mult_batch_(
  //     to_ptr,
  //     prod_inv,
  //     prod_inv_psinv,
  //     0,
  //     end_length,
  //     0,
  //     0,
  //     param_degree,
  //     to_ptr,
  //     param_primes);

  op2_ptr = reinterpret_cast<uint64_t*>(prod_inv.data_ptr<uint64_t>());
  op2_psinv_ptr =
      reinterpret_cast<uint64_t*>(prod_inv_psinv.data_ptr<uint64_t>());
  primes_ptr = reinterpret_cast<uint64_t*>(param_primes.data_ptr<uint64_t>());
#pragma omp parallel for num_threads(max_threads)
  for (int i = 0; i < param_degree * end_length; i++) {
    const int op2_idx = 0 + i / param_degree;
    const int prime_idx = i / param_degree + 0;
    const auto prime = primes_ptr[prime_idx];
    uint64_t out = fhe::mul_and_reduce_shoup(
        to_ptr[0 * param_degree + i],
        op2_ptr[op2_idx],
        op2_psinv_ptr[op2_idx],
        prime);
    if (out >= prime)
      out -= prime;
    to_ptr[0 * param_degree + i] = out;
  }
}

Tensor moddown_core_cpu(
    const Tensor& to,
    const Tensor& from,
    int64_t curr_limbs,
    int64_t level,
    int64_t alpha,
    int64_t param_degree,
    int64_t param_log_degree,
    const Tensor& hat_inverse_vec_moddown,
    const Tensor& hat_inverse_vec_shoup_moddown,
    const Tensor& prod_q_i_mod_q_j_moddown,
    const Tensor& prod_inv_moddown,
    const Tensor& prod_inv_shoup_moddown,
    const Tensor& param_primes,
    const Tensor& param_barret_ratio,
    const Tensor& param_barret_k) {
  auto res = to.clone();
  res.resize_({curr_limbs * param_degree});
  moddown_core_template(
      from,
      curr_limbs,
      level,
      alpha,
      param_degree,
      param_log_degree,
      hat_inverse_vec_moddown,
      hat_inverse_vec_shoup_moddown,
      prod_q_i_mod_q_j_moddown,
      prod_inv_moddown,
      prod_inv_shoup_moddown,
      param_primes,
      param_barret_ratio,
      param_barret_k,
      res);
  return res;
}

Tensor& moddown_core_cpu_(
    Tensor& to,
    const Tensor& from,
    int64_t curr_limbs,
    int64_t level,
    int64_t alpha,
    int64_t param_degree,
    int64_t param_log_degree,
    const Tensor& hat_inverse_vec_moddown,
    const Tensor& hat_inverse_vec_shoup_moddown,
    const Tensor& prod_q_i_mod_q_j_moddown,
    const Tensor& prod_inv_moddown,
    const Tensor& prod_inv_shoup_moddown,
    const Tensor& param_primes,
    const Tensor& param_barret_ratio,
    const Tensor& param_barret_k) {
  to.resize_({curr_limbs * param_degree});

  moddown_core_template(
      from,
      curr_limbs,
      level,
      alpha,
      param_degree,
      param_log_degree,
      hat_inverse_vec_moddown,
      hat_inverse_vec_shoup_moddown,
      prod_q_i_mod_q_j_moddown,
      prod_inv_moddown,
      prod_inv_shoup_moddown,
      param_primes,
      param_barret_ratio,
      param_barret_k,
      to);
  return to;
}

Tensor& moddown_core_cpu_out(
    const Tensor& to,
    const Tensor& from,
    int64_t curr_limbs,
    int64_t level,
    int64_t alpha,
    int64_t param_degree,
    int64_t param_log_degree,
    const Tensor& hat_inverse_vec_moddown,
    const Tensor& hat_inverse_vec_shoup_moddown,
    const Tensor& prod_q_i_mod_q_j_moddown,
    const Tensor& prod_inv_moddown,
    const Tensor& prod_inv_shoup_moddown,
    const Tensor& param_primes,
    const Tensor& param_barret_ratio,
    const Tensor& param_barret_k,
    Tensor& res) {
  res.resize_({curr_limbs * param_degree});
  moddown_core_template(
      from,
      curr_limbs,
      level,
      alpha,
      param_degree,
      param_log_degree,
      hat_inverse_vec_moddown,
      hat_inverse_vec_shoup_moddown,
      prod_q_i_mod_q_j_moddown,
      prod_inv_moddown,
      prod_inv_shoup_moddown,
      param_primes,
      param_barret_ratio,
      param_barret_k,
      res);
  return res;
}

static void moddown_cpu_template(
    // aten::moddown
    const Tensor& from,
    Tensor& workspace,
    int64_t curr_limbs,
    int64_t level,
    int64_t alpha,
    int64_t param_degree,
    int64_t param_log_degree,
    const Tensor& hat_inverse_vec_moddown,
    const Tensor& hat_inverse_vec_shoup_moddown,
    const Tensor& prod_q_i_mod_q_j_moddown,
    const Tensor& prod_inv_moddown,
    const Tensor& prod_inv_shoup_moddown,
    const Tensor& param_primes,
    const Tensor& param_barret_ratio,
    const Tensor& param_barret_k,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_power_of_roots,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    Tensor& res) {
  const int start_length = alpha;
  const int end_length = curr_limbs;

  auto hat_inverse_vec = hat_inverse_vec_moddown[0];
  auto hat_inverse_vec_psinv = hat_inverse_vec_shoup_moddown[0];
  auto workspace_ptr =
      reinterpret_cast<uint64_t*>(workspace.data_ptr<uint64_t>());
  auto from_ptr = reinterpret_cast<uint64_t*>(from.data_ptr<uint64_t>());
  auto to_ptr = reinterpret_cast<uint64_t*>(res.data_ptr<uint64_t>());

  iNTT_impl(
      from_ptr,
      workspace_ptr,
      end_length,
      start_length,
      curr_limbs,
      level,
      param_degree,
      inverse_power_of_roots_div_two,
      param_primes,
      inverse_scaled_power_of_roots_div_two);

  const_mult_batch_(
      workspace_ptr,
      hat_inverse_vec,
      hat_inverse_vec_psinv,
      level,
      alpha,
      curr_limbs,
      0,
      param_degree,
      workspace_ptr,
      param_primes);

  moddown_impl(
      workspace_ptr,
      param_degree,
      param_log_degree,
      alpha,
      start_length,
      end_length,
      param_primes,
      prod_q_i_mod_q_j_moddown,
      param_barret_ratio,
      param_barret_k,
      to_ptr);

  NTT_impl(
      to_ptr,
      to_ptr,
      0,
      end_length,
      param_degree,
      param_power_of_roots_shoup,
      param_primes,
      param_power_of_roots);

  const auto& prod_inv = prod_inv_moddown[0];
  const auto& prod_inv_psinv = prod_inv_shoup_moddown[0];

  SubInplace(to_ptr, from_ptr, end_length, param_degree, param_primes);

  NegateInplace(
      to_ptr, end_length, param_primes, param_degree, param_log_degree);

  const_mult_batch_(
      to_ptr,
      prod_inv,
      prod_inv_psinv,
      0,
      end_length,
      0,
      0,
      param_degree,
      to_ptr,
      param_primes);
}

Tensor moddown_cpu(
    const Tensor& in,
    int64_t curr_limbs,
    int64_t L,
    int64_t sizeP,
    int64_t N,
    int64_t log_degree,
    const Tensor& hat_inverse_vec_moddown,
    const Tensor& hat_inverse_vec_shoup_moddown,
    const Tensor& prod_q_i_mod_q_j_moddown,
    const Tensor& prod_inv_moddown,
    const Tensor& prod_inv_shoup_moddown,
    const Tensor& primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    const Tensor& power_of_roots_shoup,
    const Tensor& power_of_roots,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two) {
  auto from_ = in.clone();
  auto res = at::empty({curr_limbs * N}, in.options());
  auto workspace = in.clone();
  moddown_cpu_template(
      from_,
      workspace,
      curr_limbs,
      L,
      sizeP,
      N,
      log_degree,
      hat_inverse_vec_moddown,
      hat_inverse_vec_shoup_moddown,
      prod_q_i_mod_q_j_moddown,
      prod_inv_moddown,
      prod_inv_shoup_moddown,
      primes,
      barret_ratio,
      barret_k,
      power_of_roots_shoup,
      power_of_roots,
      inverse_power_of_roots_div_two,
      inverse_scaled_power_of_roots_div_two,
      res);
  return res;
}

void vec_add_mod_batch(
    uint64_t* op1_ptr,
    uint64_t* op2_ptr,
    const Tensor& primes,
    const Tensor& param_barret_ratio,
    const Tensor& param_barret_k,
    int64_t batch,
    int64_t degree,
    uint64_t* res_ptr) {
  AT_DISPATCH_V2(
      primes.scalar_type(),
      "vec_add_mod_batch_",
      AT_WRAP([&]() {
        auto primes_ptr =
            reinterpret_cast<uint64_t*>(primes.data_ptr<uint64_t>());
        auto barret_ratio_ptr = reinterpret_cast<uint64_t*>(
            param_barret_ratio.data_ptr<uint64_t>());
        auto barret_k_ptr =
            reinterpret_cast<uint64_t*>(param_barret_k.data_ptr<uint64_t>());
        const int block_dim = 256;
        const int grid_dim = degree * batch / block_dim;
        fhe::vec_add_mod_batch_(
            (int)degree,
            primes_ptr,
            barret_ratio_ptr,
            barret_k_ptr,
            op1_ptr,
            op2_ptr,
            (int)batch,
            res_ptr);
      }),
      kUInt64);
}

void vec_mod_batch(
    uint64_t* op1_ptr,
    const Tensor& primes,
    const Tensor& param_barret_ratio,
    const Tensor& param_barret_k,
    int64_t batch,
    int64_t degree,
    uint64_t* res_ptr) {
  AT_DISPATCH_V2(
      primes.scalar_type(),
      "vec_add_mod_batch_",
      AT_WRAP([&]() {
        auto primes_ptr =
            reinterpret_cast<uint64_t*>(primes.data_ptr<uint64_t>());
        auto barret_ratio_ptr = reinterpret_cast<uint64_t*>(
            param_barret_ratio.data_ptr<uint64_t>());
        auto barret_k_ptr =
            reinterpret_cast<uint64_t*>(param_barret_k.data_ptr<uint64_t>());
        const int block_dim = 256;
        const int grid_dim = degree * batch / block_dim;
        fhe::vec_mod_batch_(
            (int)degree,
            primes_ptr,
            barret_ratio_ptr,
            barret_k_ptr,
            op1_ptr,
            (int)batch,
            res_ptr);
      }),
      kUInt64);
}

void switch_modulus(
    uint64_t* ptr,
    uint64_t* res_ptr,
    const Tensor& primes,
    int64_t old_prime_index,
    int64_t batch,
    int64_t degree) {
  AT_DISPATCH_V2(
      primes.scalar_type(),
      "switch_modulus_",
      AT_WRAP([&]() {
        auto primes_ptr =
            reinterpret_cast<uint64_t*>(primes.data_ptr<uint64_t>());
        const int block_dim = 256;
        const int grid_dim = degree * batch / block_dim;
        fhe::switch_modulus_(
            (int)degree, batch, old_prime_index, primes_ptr, ptr, res_ptr);
      }),
      kUInt64);
}

static void drop_last_element_scale_template(
    // aten::drop_last_element_and_scale

    const Tensor& from,
    int64_t curr_limbs,
    int64_t l,
    int64_t level,
    int64_t param_degree,
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
  auto to_ptr = reinterpret_cast<uint64_t*>(res.data_ptr<uint64_t>());
  iNTT_impl(
      from_ptr,
      from_ptr,
      end_length,
      1,
      curr_limbs,
      level,
      param_degree,
      inverse_power_of_roots_div_two,
      param_primes,
      inverse_scaled_power_of_roots_div_two);

  auto ptr = from_ptr + param_degree * end_length;

  switch_modulus(
      ptr, to_ptr, param_primes, curr_limbs - 1, curr_limbs - 1, param_degree);

  int start_op2_idx = (level - curr_limbs + l) * (level - 1);

  const_mult_batch_(
      to_ptr,
      qlql_inv_mod_ql_div_ql_mod_q,
      qlql_inv_mod_ql_div_ql_mod_q_shoup,
      0,
      curr_limbs - 1,
      0,
      start_op2_idx,
      param_degree,
      to_ptr,
      param_primes);
  NTT_impl(
      to_ptr,
      to_ptr,
      0,
      end_length,
      param_degree,
      param_power_of_roots_shoup,
      param_primes,
      param_power_of_roots);

  start_op2_idx = (curr_limbs - 1) * (level);
  const_mult_batch_(
      from_ptr,
      q_inv_mod_q,
      q_inv_mod_q_shoup,
      0,
      end_length,
      0,
      start_op2_idx,
      param_degree,
      from_ptr,
      param_primes);

  vec_add_mod_batch(
      to_ptr,
      from_ptr,
      param_primes,
      param_barret_ratio,
      param_barret_k,
      end_length,
      param_degree,
      to_ptr);
}

Tensor drop_last_element_scale_cpu(
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
  auto res = to.clone();
  res.resize_({(curr_limbs - 1) * N});

  drop_last_element_scale_template(
      from,
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

Tensor& drop_last_element_scale_cpu_(
    Tensor& to,
    const Tensor& from,
    int64_t curr_limbs,
    int64_t l,
    int64_t level,
    int64_t param_degree,
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
  to.resize_({(curr_limbs - 1) * param_degree});

  drop_last_element_scale_template(
      from,
      curr_limbs,
      l,
      level,
      param_degree,
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
      to);

  return to;
}

Tensor& drop_last_element_scale_cpu_out(
    const Tensor& to,
    const Tensor& from,
    int64_t curr_limbs,
    int64_t l,
    int64_t level,
    int64_t param_degree,
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
  res.resize_({(curr_limbs - 1) * param_degree});

  drop_last_element_scale_template(
      from,
      curr_limbs,
      l,
      level,
      param_degree,
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

static void rescale_template(
    const Tensor& from,
    int64_t curr_limbs,
    int64_t level,
    int64_t param_degree,
    const Tensor& param_primes,
    const Tensor& param_barret_ratio,
    const Tensor& param_barret_k,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_power_of_roots,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& q_inv_mod_q,
    const Tensor& q_inv_mod_q_shoup,
    Tensor& res) {
  const int end_length = curr_limbs - 1;
  auto from_ptr = reinterpret_cast<uint64_t*>(from.data_ptr<uint64_t>());
  auto to_ptr = reinterpret_cast<uint64_t*>(res.data_ptr<uint64_t>());
  iNTT_impl(
      from_ptr,
      from_ptr,
      end_length,
      1,
      curr_limbs,
      level,
      param_degree,
      inverse_power_of_roots_div_two,
      param_primes,
      inverse_scaled_power_of_roots_div_two);

  auto ptr = from_ptr + param_degree * end_length;

  vec_mod_batch(
      ptr,
      param_primes,
      param_barret_ratio,
      param_barret_k,
      end_length,
      param_degree,
      to_ptr);

  NTT_impl(
      to_ptr,
      to_ptr,
      0,
      end_length,
      param_degree,
      param_power_of_roots_shoup,
      param_primes,
      param_power_of_roots);

  SubInplace(from_ptr, to_ptr, end_length, param_degree, param_primes);

  int start_op2_idx = (curr_limbs - 1) * (level);
  const_mult_batch_(
      from_ptr,
      q_inv_mod_q,
      q_inv_mod_q_shoup,
      0,
      end_length,
      0,
      start_op2_idx,
      param_degree,
      to_ptr,
      param_primes);
}

Tensor rescale_cpu(
    const Tensor& to,
    const Tensor& from,
    int64_t curr_limbs,
    int64_t level,
    int64_t param_degree,
    const Tensor& param_primes,
    const Tensor& param_barret_ratio,
    const Tensor& param_barret_k,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_power_of_roots,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& q_inv_mod_q,
    const Tensor& q_inv_mod_q_shoup) {
  auto res = to.clone();
  res.resize_({(curr_limbs - 1) * param_degree});

  rescale_template(
      from,
      curr_limbs,
      level,
      param_degree,
      param_primes,
      param_barret_ratio,
      param_barret_k,
      param_power_of_roots_shoup,
      param_power_of_roots,
      inverse_power_of_roots_div_two,
      inverse_scaled_power_of_roots_div_two,
      q_inv_mod_q,
      q_inv_mod_q_shoup,
      res);

  return res;
}

Tensor& rescale_cpu_(
    Tensor& to,
    const Tensor& from,
    int64_t curr_limbs,
    int64_t level,
    int64_t param_degree,
    const Tensor& param_primes,
    const Tensor& param_barret_ratio,
    const Tensor& param_barret_k,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_power_of_roots,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& q_inv_mod_q,
    const Tensor& q_inv_mod_q_shoup) {
  to.resize_({(curr_limbs - 1) * param_degree});

  rescale_template(
      from,
      curr_limbs,
      level,
      param_degree,
      param_primes,
      param_barret_ratio,
      param_barret_k,
      param_power_of_roots_shoup,
      param_power_of_roots,
      inverse_power_of_roots_div_two,
      inverse_scaled_power_of_roots_div_two,
      q_inv_mod_q,
      q_inv_mod_q_shoup,
      to);

  return to;
}

Tensor& rescale_cpu_out(
    const Tensor& to,
    const Tensor& from,
    int64_t curr_limbs,
    int64_t level,
    int64_t param_degree,
    const Tensor& param_primes,
    const Tensor& param_barret_ratio,
    const Tensor& param_barret_k,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_power_of_roots,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& q_inv_mod_q,
    const Tensor& q_inv_mod_q_shoup,
    Tensor& res) {
  res.resize_({(curr_limbs - 1) * param_degree});

  rescale_template(
      from,
      curr_limbs,
      level,
      param_degree,
      param_primes,
      param_barret_ratio,
      param_barret_k,
      param_power_of_roots_shoup,
      param_power_of_roots,
      inverse_power_of_roots_div_two,
      inverse_scaled_power_of_roots_div_two,
      q_inv_mod_q,
      q_inv_mod_q_shoup,
      res);

  return res;
}

} // namespace at::native
