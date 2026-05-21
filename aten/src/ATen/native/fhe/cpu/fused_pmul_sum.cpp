#include <ATen/Dispatch_v2.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/interned_strings.h>
#include <ATen/ops/copy.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>
#include <omp.h>

#include "ATen/native/fhe/cpu/Utils.h"
#include "ATen/native/fhe/cpu/arithmetic.h"

#pragma clang diagnostic ignored "-Wmissing-prototypes"

namespace at::native {

Tensor batched_pairwise_mac_cpu(
    const Tensor& cipher,
    const Tensor& plaintext,
    const Tensor& param_primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    int64_t number_batches,
    int64_t num_cipher,
    int64_t cur_limbs,
    int64_t N) {
  auto res = at::empty({2, number_batches, cur_limbs, N}, cipher.options());

  auto* res_ptr = res.data_ptr<uint64_t>();
  const auto* cipher_ptr = cipher.data_ptr<uint64_t>();
  const auto* plain_ptr = plaintext.data_ptr<uint64_t>();
  const auto* mods = param_primes.data_ptr<uint64_t>();
  const auto* ratios = barret_ratio.data_ptr<uint64_t>();
  const auto* ks = barret_k.data_ptr<uint64_t>();

  const int64_t L_CTN = cipher.size(2) * N;
  const int64_t BL_CTN = cipher.size(1) * L_CTN;
  const int64_t L_PTN = plaintext.size(2) * N;

  const int max_threads = omp_get_max_threads();
#pragma omp parallel for collapse(3) schedule(static) num_threads(max_threads)
  for (int64_t batch_id = 0; batch_id < number_batches; ++batch_id) {
    for (int64_t limb = 0; limb < cur_limbs; ++limb) {
      for (int64_t n = 0; n < N; ++n) {
        __uint128_t sum_bx{0};
        __uint128_t sum_ax{0};
        for (int64_t i = 0; i < num_cipher; ++i) {
          const int64_t plain_off =
              (batch_id * num_cipher + i) * L_PTN + limb * N + n;
          const int64_t cipher_off = i * L_CTN + limb * N + n;
          const uint64_t plain_val = plain_ptr[plain_off];
          const uint64_t cipher_val_bx = cipher_ptr[cipher_off];
          const uint64_t cipher_val_ax = cipher_ptr[cipher_off + BL_CTN];
          sum_bx += static_cast<__uint128_t>(cipher_val_bx) * plain_val;
          sum_ax += static_cast<__uint128_t>(cipher_val_ax) * plain_val;
        }

        const uint64_t mod = mods[limb];
        const uint64_t ratio = ratios[limb];
        const unsigned k = static_cast<unsigned>(ks[limb]);
        res_ptr[batch_id * cur_limbs * N + limb * N + n] =
            fhe::barret_reduction_128_64(sum_bx, mod, ratio, k);
        res_ptr[number_batches * cur_limbs * N + batch_id * cur_limbs * N +
                limb * N + n] =
            fhe::barret_reduction_128_64(sum_ax, mod, ratio, k);
      }
    }
  }

  return res;
}

Tensor fused_broadcast_mac_cpu(
    const Tensor& cipher,
    const Tensor& plaintext,
    const Tensor& param_primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    int64_t num_plain,
    int64_t cur_limbs,
    int64_t N) {
  auto res = at::empty({2, cur_limbs, N}, cipher.options());

  auto* res_ptr = res.data_ptr<uint64_t>();
  const auto* cipher_ptr = cipher.data_ptr<uint64_t>();
  const auto* plain_ptr = plaintext.data_ptr<uint64_t>();
  const auto* mods = param_primes.data_ptr<uint64_t>();
  const auto* ratios = barret_ratio.data_ptr<uint64_t>();
  const auto* ks = barret_k.data_ptr<uint64_t>();

  const int64_t L_CTN = cipher.size(1) * N;
  const int64_t L_PTN = plaintext.size(2) * N;

  const int max_threads = omp_get_max_threads();
#pragma omp parallel for collapse(2) schedule(static) num_threads(max_threads)
  for (int64_t limb = 0; limb < cur_limbs; ++limb) {
    for (int64_t n = 0; n < N; ++n) {
      __uint128_t sum_bx{0};
      __uint128_t sum_ax{0};
      const uint64_t cipher_bx = cipher_ptr[limb * N + n];
      const uint64_t cipher_ax = cipher_ptr[L_CTN + limb * N + n];

      for (int64_t i = 0; i < num_plain; ++i) {
        const uint64_t plain_val = plain_ptr[i * L_PTN + limb * N + n];
        sum_bx += static_cast<__uint128_t>(cipher_bx) * plain_val;
        sum_ax += static_cast<__uint128_t>(cipher_ax) * plain_val;
      }

      const uint64_t mod = mods[limb];
      const uint64_t ratio = ratios[limb];
      const unsigned k = static_cast<unsigned>(ks[limb]);
      res_ptr[limb * N + n] =
          fhe::barret_reduction_128_64(sum_bx, mod, ratio, k);
      res_ptr[cur_limbs * N + limb * N + n] =
          fhe::barret_reduction_128_64(sum_ax, mod, ratio, k);
    }
  }

  return res;
}

Tensor scalar_weighted_acc_cpu(
    const Tensor& cipher,
    const Tensor& scalars,
    const Tensor& param_primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    int64_t num_cipher,
    int64_t cur_limbs,
    int64_t N) {
  auto res = at::empty({2, cur_limbs, N}, cipher.options());

  auto* res_ptr = res.data_ptr<uint64_t>();
  const auto* cipher_ptr = cipher.data_ptr<uint64_t>();
  const auto* scalar_ptr = scalars.data_ptr<uint64_t>();
  const auto* mods = param_primes.data_ptr<uint64_t>();
  const auto* ratios = barret_ratio.data_ptr<uint64_t>();
  const auto* ks = barret_k.data_ptr<uint64_t>();

  const int64_t L_CTN = cipher.size(2) * N;
  const int64_t BL_CTN = cipher.size(1) * L_CTN;

  const int max_threads = omp_get_max_threads();
#pragma omp parallel for collapse(2) schedule(static) num_threads(max_threads)
  for (int64_t limb = 0; limb < cur_limbs; ++limb) {
    for (int64_t n = 0; n < N; ++n) {
      __uint128_t sum_bx{0};
      __uint128_t sum_ax{0};
      for (int64_t i = 0; i < num_cipher; ++i) {
        const uint64_t scalar_val = scalar_ptr[i * cur_limbs + limb];
        const int64_t cipher_off = i * L_CTN + limb * N + n;
        const uint64_t cipher_val_bx = cipher_ptr[cipher_off];
        const uint64_t cipher_val_ax = cipher_ptr[cipher_off + BL_CTN];
        sum_bx += static_cast<__uint128_t>(cipher_val_bx) * scalar_val;
        sum_ax += static_cast<__uint128_t>(cipher_val_ax) * scalar_val;
      }

      const uint64_t mod = mods[limb];
      const uint64_t ratio = ratios[limb];
      const unsigned k = static_cast<unsigned>(ks[limb]);
      res_ptr[limb * N + n] =
          fhe::barret_reduction_128_64(sum_bx, mod, ratio, k);
      res_ptr[cur_limbs * N + limb * N + n] =
          fhe::barret_reduction_128_64(sum_ax, mod, ratio, k);
    }
  }

  return res;
}

Tensor grouped_scalar_weighted_acc_cpu(
    const Tensor& cipher,
    const Tensor& scalars,
    const Tensor& param_primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    int64_t num_groups,
    int64_t num_cipher,
    int64_t cur_limbs,
    int64_t N,
    int64_t strategy) {
  (void)strategy;
  auto res = at::empty({2, num_groups, cur_limbs, N}, cipher.options());

  auto* res_ptr = res.data_ptr<uint64_t>();
  const auto* cipher_ptr = cipher.data_ptr<uint64_t>();
  const auto* scalar_ptr = scalars.data_ptr<uint64_t>();
  const auto* mods = param_primes.data_ptr<uint64_t>();
  const auto* ratios = barret_ratio.data_ptr<uint64_t>();
  const auto* ks = barret_k.data_ptr<uint64_t>();

  const int64_t L_CTN = cipher.size(2) * N;
  const int64_t BL_CTN = cipher.size(1) * L_CTN;

  const int max_threads = omp_get_max_threads();
#pragma omp parallel for collapse(3) schedule(static) num_threads(max_threads)
  for (int64_t group = 0; group < num_groups; ++group) {
    for (int64_t limb = 0; limb < cur_limbs; ++limb) {
      for (int64_t n = 0; n < N; ++n) {
        __uint128_t sum_bx{0};
        __uint128_t sum_ax{0};
        for (int64_t i = 0; i < num_cipher; ++i) {
          const uint64_t scalar_val =
              scalar_ptr[(group * num_cipher + i) * cur_limbs + limb];
          const int64_t cipher_off = i * L_CTN + limb * N + n;
          const uint64_t cipher_val_bx = cipher_ptr[cipher_off];
          const uint64_t cipher_val_ax = cipher_ptr[cipher_off + BL_CTN];
          sum_bx += static_cast<__uint128_t>(cipher_val_bx) * scalar_val;
          sum_ax += static_cast<__uint128_t>(cipher_val_ax) * scalar_val;
        }

        const uint64_t mod = mods[limb];
        const uint64_t ratio = ratios[limb];
        const unsigned k = static_cast<unsigned>(ks[limb]);
        const int64_t out_off = group * cur_limbs * N + limb * N + n;
        res_ptr[out_off] =
            fhe::barret_reduction_128_64(sum_bx, mod, ratio, k);
        res_ptr[num_groups * cur_limbs * N + out_off] =
            fhe::barret_reduction_128_64(sum_ax, mod, ratio, k);
      }
    }
  }

  return res;
}

Tensor cpmul_broadcast_pt_cpu(
    const Tensor& cipher,
    const Tensor& plaintext,
    const Tensor& param_primes,
    const Tensor& barret_mu,
    int64_t num_cipher,
    int64_t cur_limbs,
    int64_t N) {
  auto res = at::empty({2, num_cipher, cur_limbs, N}, cipher.options());

  auto* res_ptr = res.data_ptr<uint64_t>();
  const auto* cipher_ptr = cipher.data_ptr<uint64_t>();
  const auto* plain_ptr = plaintext.data_ptr<uint64_t>();
  const auto* mods = param_primes.data_ptr<uint64_t>();
  const auto* mu = barret_mu.data_ptr<uint64_t>();

  const int64_t L_CTN = cipher.size(2) * N;
  const int64_t BL_CTN = cipher.size(1) * L_CTN;

  const int max_threads = omp_get_max_threads();
#pragma omp parallel for collapse(3) schedule(static) num_threads(max_threads)
  for (int64_t batch_id = 0; batch_id < num_cipher; ++batch_id) {
    for (int64_t limb = 0; limb < cur_limbs; ++limb) {
      for (int64_t n = 0; n < N; ++n) {
        const uint64_t mod = mods[limb];
        const uint64_t mu0 = mu[limb * 2];
        const uint64_t mu1 = mu[limb * 2 + 1];
        const uint64_t ptx_val = plain_ptr[limb * N + n];

        const int64_t off = batch_id * L_CTN + limb * N + n;
        const uint64_t cipher_bx = cipher_ptr[off];
        const uint64_t cipher_ax = cipher_ptr[off + BL_CTN];
        res_ptr[off] = fhe::mul_mod(cipher_bx, ptx_val, mod, mu0, mu1);
        res_ptr[off + BL_CTN] = fhe::mul_mod(cipher_ax, ptx_val, mod, mu0, mu1);
      }
    }
  }

  return res;
}

Tensor cpmul_broadcast_cipher_cpu(
    const Tensor& cipher,
    const Tensor& plaintext,
    const Tensor& param_primes,
    const Tensor& barret_mu,
    int64_t num_cipher,
    int64_t cur_limbs,
    int64_t N) {
  auto res = at::empty({2, num_cipher, cur_limbs, N}, cipher.options());

  auto* res_ptr = res.data_ptr<uint64_t>();
  const auto* cipher_ptr = cipher.data_ptr<uint64_t>();
  const auto* plain_ptr = plaintext.data_ptr<uint64_t>();
  const auto* mods = param_primes.data_ptr<uint64_t>();
  const auto* mu = barret_mu.data_ptr<uint64_t>();

  const int64_t L_CTN = cipher.size(2) * N;
  const int64_t L_PTN = plaintext.size(2) * N;

  const int max_threads = omp_get_max_threads();
#pragma omp parallel for collapse(3) schedule(static) num_threads(max_threads)
  for (int64_t batch_id = 0; batch_id < num_cipher; ++batch_id) {
    for (int64_t limb = 0; limb < cur_limbs; ++limb) {
      for (int64_t n = 0; n < N; ++n) {
        const uint64_t mod = mods[limb];
        const uint64_t mu0 = mu[limb * 2];
        const uint64_t mu1 = mu[limb * 2 + 1];

        const uint64_t cipher_bx = cipher_ptr[limb * N + n];
        const uint64_t cipher_ax = cipher_ptr[limb * N + n + L_CTN];
        const int64_t out_off = batch_id * L_PTN + limb * N + n;
        const uint64_t ptx_val = plain_ptr[out_off];

        res_ptr[out_off] = fhe::mul_mod(cipher_bx, ptx_val, mod, mu0, mu1);
        res_ptr[out_off + num_cipher * cur_limbs * N] =
            fhe::mul_mod(cipher_ax, ptx_val, mod, mu0, mu1);
      }
    }
  }

  return res;
}

} // namespace at::native
