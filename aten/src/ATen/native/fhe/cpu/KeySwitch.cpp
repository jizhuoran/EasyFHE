#include "ATen/native/fhe/cpu/KeySwitch.h"
#include <ATen/Dispatch_v2.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/interned_strings.h>
#include <ATen/ops/copy.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>
#include <omp.h>
#include <iostream>
#include "ATen/native/fhe/cpu/NttImpl.h"
#pragma clang diagnostic ignored "-Wmissing-prototypes"

namespace fhe {

__uint128_t accumulate_in_modup(
    const uint64_t* ptr,
    const int degree,
    const uint64_t* hat_mod_end,
    const int start_length, // sizeP
    const int degree_idx,
    const int hat_mod_end_idx) {
  __uint128_t accum{0};

  for (int i = 0; i < start_length; i++) {
    const uint64_t op2 = hat_mod_end[hat_mod_end_idx * start_length + i];
    accum += static_cast<uint128_t>(ptr[i * degree + degree_idx]) * op2;
  }
  return accum;
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
    const int out_prime_idx = i / degree_;
    const auto prime = d_primes[out_prime_idx];
    const auto barret_ratio = d_barret_ratio[out_prime_idx];
    const auto barret_k = d_barret_k[out_prime_idx];
    to[i] =
        barret_reduction_64_64(op1[i] + op2[i], prime, barret_ratio, barret_k);
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
  to[i] = barret_reduction_64_64(op1[op1_idx], prime, barret_ratio, barret_k);

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

    uint64_t out = fhe::mul_and_reduce_shoup(
        op1[start_op1_idx * degree + i],
        op2[op2_idx],
        op2_psinv[op2_idx],
        prime);

    if (out >= prime)
      out -= prime;
    to[start_op1_idx * degree + i] = out;
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
        const_mult_batch(
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

} // namespace at::native
