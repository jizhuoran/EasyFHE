#include <ATen/Dispatch_v2.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/interned_strings.h>
#include <ATen/ops/copy.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include "ATen/native/fhe/cpu/CommonOperation.h"
#include "ATen/native/fhe/cpu/NttImpl.h"
#include "ATen/native/fhe/cpu/Utils.h"

namespace fhe {
void sampleGauss(
    uint64_t* res,
    uint64_t* pVec,
    uint64_t* qVec,
    int64_t l,
    int64_t k,
    int64_t N,
    int64_t logN) {
  uint64_t seed = (uint64_t)std::random_device{}();
  std::mt19937_64 gen(seed);
  std::normal_distribution<double> dist(0.0, 3.2);
  for (long i = 0; i < N; i += 2) {
    long g1 = static_cast<long>(std::lround(dist(gen)));
    long g2 = static_cast<long>(std::lround(dist(gen)));

    for (long j = 0; j < l; ++j) {
      uint64_t* resj = res + (j << logN);
      if (g1 >= 0)
        resj[i] = static_cast<uint64_t>(g1);
      else
        resj[i] = qVec[j] - static_cast<uint64_t>(-g1);
      if (g2 >= 0)
        resj[i + 1] = static_cast<uint64_t>(g2);
      else
        resj[i + 1] = qVec[j] - static_cast<uint64_t>(-g2);
    }
  }
}

void mymul(
    uint64_t* res,
    uint64_t* pVec,
    uint64_t* qVec,
    uint64_t* a,
    uint64_t* b,
    int64_t l,
    int64_t k,
    int64_t logN,
    const uint64_t* barret_ratio_ptr,
    const uint64_t* barret_k_ptr,
    uint64_t* primes_ptr) {
  int64_t N = 1 << logN;
  for (int64_t i = 0; i < l; ++i) {
    uint64_t* ai = a + (i << logN);
    uint64_t* bi = b + (i << logN);
    uint64_t* resi = res + (i << logN);
    for (int64_t j = 0; j < N; ++j) {
      resi[j] = mul_mod_barrett(
          ai[j], bi[j], qVec[i], barret_ratio_ptr[i], barret_k_ptr[i]);
    }
  }
}

void addAndEqual(
    uint64_t* a,
    uint64_t* b,
    uint64_t* pVec,
    uint64_t* qVec,
    int64_t l,
    int64_t k,
    int64_t N,
    int64_t logN) {
  for (int64_t i = 0; i < l; ++i) {
    uint64_t* ai = a + (i << logN);
    uint64_t* bi = b + (i << logN);
    for (int64_t j = 0; j < N; ++j) {
      ai[j] = add_mod(ai[j], bi[j], qVec[i]);
    }
  }
}

} // namespace fhe
namespace at::native {
static void encryp_template_cpu(
    Tensor& ax,
    Tensor& bx,
    Tensor& vx,
    Tensor& ex,
    const Tensor& ptx,
    const Tensor& pk0, // key.ax
    const Tensor& pk1, // key.bx
    int64_t l,
    int64_t logn,
    int64_t n,
    int64_t nh,
    const Tensor& moduliP_scalar,
    const Tensor& moduliQ_scalar,
    const Tensor& primes,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    const Tensor& power_of_roots_shoup,
    const Tensor& power_of_roots) {
  auto ax_ptr = reinterpret_cast<uint64_t*>(ax.data_ptr<uint64_t>());
  auto bx_ptr = reinterpret_cast<uint64_t*>(bx.data_ptr<uint64_t>());
  auto vx_ptr = reinterpret_cast<uint64_t*>(vx.data_ptr<uint64_t>());
  auto ex_ptr = reinterpret_cast<uint64_t*>(ex.data_ptr<uint64_t>());
  auto ptx_ptr = reinterpret_cast<uint64_t*>(ptx.data_ptr<uint64_t>());
  auto pk0_ptr = reinterpret_cast<uint64_t*>(pk0.data_ptr<uint64_t>());
  auto pk1_ptr = reinterpret_cast<uint64_t*>(pk1.data_ptr<uint64_t>());
  auto moduliP_scalar_ptr =
      reinterpret_cast<uint64_t*>(moduliP_scalar.data_ptr<uint64_t>());
  auto moduliQ_scalar_ptr =
      reinterpret_cast<uint64_t*>(moduliQ_scalar.data_ptr<uint64_t>());
  auto primes_ptr = reinterpret_cast<uint64_t*>(primes.data_ptr<uint64_t>());
  auto barret_ratio_ptr =
      reinterpret_cast<uint64_t*>(barret_ratio.data_ptr<uint64_t>());
  auto barret_k_ptr =
      reinterpret_cast<uint64_t*>(barret_k.data_ptr<uint64_t>());
  int64_t k = 0;
  fhe::sampleGauss(
      vx_ptr, moduliP_scalar_ptr, moduliQ_scalar_ptr, l, k, n, logn);
  NTT_impl(
      vx_ptr, k, l, 1 << logn, power_of_roots_shoup, primes, power_of_roots);
  fhe::mymul(
      bx_ptr,
      moduliP_scalar_ptr,
      moduliQ_scalar_ptr,
      vx_ptr,
      pk0_ptr,
      l,
      k,
      logn,
      barret_ratio_ptr,
      barret_k_ptr,
      primes_ptr);
  fhe::sampleGauss(
      ex_ptr, moduliP_scalar_ptr, moduliQ_scalar_ptr, l, k, n, logn);
  NTT_impl(
      ex_ptr, k, l, 1 << logn, power_of_roots_shoup, primes, power_of_roots);
  fhe::addAndEqual(
      bx_ptr, ex_ptr, moduliP_scalar_ptr, moduliQ_scalar_ptr, l, k, n, logn);
  fhe::mymul(
      ax_ptr,
      moduliP_scalar_ptr,
      moduliQ_scalar_ptr,
      vx_ptr,
      pk1_ptr,
      l,
      k,
      logn,
      barret_ratio_ptr,
      barret_k_ptr,
      primes_ptr);

  fhe::sampleGauss(
      ex_ptr, moduliP_scalar_ptr, moduliQ_scalar_ptr, l, k, n, logn);

  NTT_impl(
      ex_ptr, k, l, 1 << logn, power_of_roots_shoup, primes, power_of_roots);

  fhe::addAndEqual(
      ax_ptr, ex_ptr, moduliP_scalar_ptr, moduliQ_scalar_ptr, l, k, n, logn);

  fhe::addAndEqual(
      bx_ptr, ptx_ptr, moduliP_scalar_ptr, moduliQ_scalar_ptr, l, k, n, logn);
}

std::vector<Tensor> encrypt_cpu(
    const Tensor& ptx,
    const Tensor& pk0,
    const Tensor& pk1,
    int64_t l,
    int64_t logn,
    int64_t nh,
    const Tensor& moduliP_scalar,
    const Tensor& moduliQ_scalar,
    const Tensor& primes,
    const Tensor& max_int_diffs,
    const Tensor& barret_ratio,
    const Tensor& barret_k,
    const Tensor& power_of_roots_shoup,
    const Tensor& power_of_roots) {
  int64_t size = l << logn;
  Tensor ax = at::zeros({size}, at::kUInt64);
  Tensor bx = at::zeros({size}, at::kUInt64);
  Tensor vx = at::zeros({size}, at::kUInt64);
  Tensor ex = at::zeros({size}, at::kUInt64);
  int64_t n = 1 << logn;
  encryp_template_cpu(
      ax,
      bx,
      vx,
      ex,
      ptx,
      pk0,
      pk1,
      l,
      logn,
      n,
      nh,
      moduliP_scalar,
      moduliQ_scalar,
      primes,
      barret_ratio,
      barret_k,
      power_of_roots_shoup,
      power_of_roots);
  return {bx, ax};
}
} // namespace at::native
