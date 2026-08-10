#include <ATen/Dispatch_v2.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/interned_strings.h>
#include <ATen/ops/copy.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>
#include <omp.h>

#include <cstring>

#include "ATen/native/fhe/cpu/CommonOperation.h"
#include "ATen/native/fhe/cpu/Utils.h"

#pragma clang diagnostic ignored "-Wmissing-prototypes"

namespace fhe {

void mul_by_monomial_impl(
    uint64_t* out,
    const uint64_t* in,
    const uint64_t* q_vec,
    int64_t l,
    int64_t N,
    int64_t M,
    int64_t monomial_deg) {
  int64_t shift = monomial_deg % M;

  const int max_threads = omp_get_max_threads();
  if (shift < N) {
#pragma omp parallel for collapse(2) schedule(static) num_threads(max_threads)
    for (int64_t row = 0; row < l; ++row) {
      for (int64_t x = 0; x < N; ++x) {
        if (x < shift) {
          const auto in_val = in[row * N + x + (N - shift)];
          out[row * N + x] = in_val == 0 ? 0 : q_vec[row] - in_val;
        } else {
          out[row * N + x] = in[row * N + x - shift];
        }
      }
    }
  } else {
    shift %= N;
#pragma omp parallel for collapse(2) schedule(static) num_threads(max_threads)
    for (int64_t row = 0; row < l; ++row) {
      for (int64_t x = 0; x < N; ++x) {
        if (x < shift) {
          out[row * N + x] = in[row * N + x + (N - shift)];
        } else {
          const auto in_val = in[row * N + x - shift];
          out[row * N + x] = in_val == 0 ? 0 : q_vec[row] - in_val;
        }
      }
    }
  }
}

} // namespace fhe

namespace at::native {

static void mul_by_monomial_inplace_template(
    Tensor& res,
    int64_t l,
    int64_t N,
    int64_t M,
    int64_t monomialDeg,
    int64_t level,
    const Tensor& param_primes,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_power_of_roots) {
  (void)level;
  const auto num_cv = res.sizes()[0];
  const auto num_cipher = res.sizes()[1];
  const auto L = res.sizes()[2];
  const auto LN = L * N;
  const auto BLN = LN * num_cipher;

  auto* res_ptr_ = res.data_ptr<uint64_t>();

  iNTT_impl(
      res_ptr_,
      res_ptr_,
      l,
      N,
      L,
      L,
      num_cv,
      num_cipher,
      param_primes.data_ptr<uint64_t>(),
      inverse_power_of_roots_div_two.data_ptr<uint64_t>(),
      inverse_scaled_power_of_roots_div_two.data_ptr<uint64_t>());

  for (int64_t cv_id = 0; cv_id < num_cv; ++cv_id) {
    for (int64_t batch = 0; batch < num_cipher; ++batch) {
      auto* res_ptr = res_ptr_ + cv_id * BLN + batch * LN;
      Tensor temp = at::empty({l, N}, res.options());
      auto* temp_ptr = temp.data_ptr<uint64_t>();

      fhe::mul_by_monomial_impl(
          temp_ptr,
          res_ptr,
          param_primes.data_ptr<uint64_t>(),
          l,
          N,
          M,
          monomialDeg);

      std::memcpy(res_ptr, temp_ptr, static_cast<size_t>(l * N) * sizeof(uint64_t));
    }
  }

  NTT_impl(
      res_ptr_,
      l,
      N,
      L,
      num_cv,
      num_cipher,
      param_primes.data_ptr<uint64_t>(),
      param_power_of_roots_shoup.data_ptr<uint64_t>(),
      param_power_of_roots.data_ptr<uint64_t>());
}

Tensor mul_by_monomial_cpu(
    const Tensor& res,
    int64_t l,
    int64_t N,
    int64_t M,
    int64_t monomialDeg,
    int64_t level,
    const Tensor& param_primes,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_power_of_roots) {
  (void)l;
  (void)N;
  (void)M;
  (void)monomialDeg;
  (void)level;
  (void)param_primes;
  (void)inverse_power_of_roots_div_two;
  (void)inverse_scaled_power_of_roots_div_two;
  (void)param_power_of_roots_shoup;
  (void)param_power_of_roots;
  TORCH_INTERNAL_ASSERT(false, "mul_by_monomial_cpu only supports inplace operation");
  return res;
}

Tensor& mul_by_monomial_cpu_(
    Tensor& res,
    int64_t l,
    int64_t N,
    int64_t M,
    int64_t monomialDeg,
    int64_t level,
    const Tensor& param_primes,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_power_of_roots) {
  TORCH_INTERNAL_ASSERT(res.dim() == 4);
  TORCH_INTERNAL_ASSERT(res.sizes()[0] > 0);

  mul_by_monomial_inplace_template(
      res,
      l,
      N,
      M,
      monomialDeg,
      level,
      param_primes,
      inverse_power_of_roots_div_two,
      inverse_scaled_power_of_roots_div_two,
      param_power_of_roots_shoup,
      param_power_of_roots);
  return res;
}

Tensor& mul_by_monomial_cpu_out(
    const Tensor& res,
    int64_t l,
    int64_t N,
    int64_t M,
    int64_t monomialDeg,
    int64_t level,
    const Tensor& param_primes,
    const Tensor& inverse_power_of_roots_div_two,
    const Tensor& inverse_scaled_power_of_roots_div_two,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_power_of_roots,
    Tensor& out) {
  (void)res;
  (void)l;
  (void)N;
  (void)M;
  (void)monomialDeg;
  (void)level;
  (void)param_primes;
  (void)inverse_power_of_roots_div_two;
  (void)inverse_scaled_power_of_roots_div_two;
  (void)param_power_of_roots_shoup;
  (void)param_power_of_roots;
  TORCH_INTERNAL_ASSERT(false, "Not implemented");
  return out;
}

} // namespace at::native
