#include <ATen/Dispatch_v2.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/zeros.h>
#include <c10/util/Exception.h>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <vector>
#include <omp.h>

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

uint32_t reverse_bits(uint32_t value, uint32_t width) {
  uint32_t result = 0;
  for (uint32_t i = 0; i < width; ++i) {
    result = (result << 1) | (value & 1U);
    value >>= 1;
  }
  return result;
}

uint32_t bit_width(uint64_t value) {
  uint32_t width = 0;
  while (value != 0) {
    ++width;
    value >>= 1;
  }
  return width;
}

uint64_t mod_mul(uint64_t lhs, uint64_t rhs, uint64_t modulus) {
  return static_cast<uint64_t>(
      (static_cast<unsigned __int128>(lhs) * rhs) % modulus);
}

uint64_t mod_add(uint64_t lhs, uint64_t rhs, uint64_t modulus) {
  const uint64_t sum = lhs + rhs;
  if (sum < lhs || sum >= modulus) {
    return sum - modulus;
  }
  return sum;
}

uint64_t mod_pow(uint64_t base, uint64_t exponent, uint64_t modulus) {
  uint64_t result = 1 % modulus;
  while (exponent > 0) {
    if ((exponent & 1U) != 0) {
      result = mod_mul(result, base, modulus);
    }
    exponent >>= 1;
    if (exponent != 0) {
      base = mod_mul(base, base, modulus);
    }
  }
  return result;
}

std::vector<uint64_t> root_table(uint64_t root, int64_t ring_dim, uint64_t modulus) {
  std::vector<uint64_t> table(static_cast<size_t>(ring_dim));
  uint64_t x = 1;
  const uint32_t msb = bit_width(static_cast<uint64_t>(ring_dim - 1));
  for (int64_t i = 0; i < ring_dim; ++i) {
    table[reverse_bits(static_cast<uint32_t>(i), msb)] = x;
    x = mod_mul(x, root, modulus);
  }
  return table;
}

void inverse_ntt_from_eval(
    const uint64_t* values,
    uint64_t* out,
    int64_t ring_dim,
    uint64_t root,
    uint64_t modulus) {
  std::copy(values, values + ring_dim, out);

  const uint64_t root_inv = mod_pow(root, modulus - 2, modulus);
  const std::vector<uint64_t> root_inv_table = root_table(root_inv, ring_dim, modulus);

  int64_t t = 1;
  int64_t logt1 = 1;
  int64_t m = ring_dim >> 1;
  while (m >= 1) {
    for (int64_t i = 0; i < m; ++i) {
      const int64_t j1 = i << logt1;
      const int64_t j2 = j1 + t;
      const uint64_t omega = root_inv_table[static_cast<size_t>(m + i)];
      for (int64_t index_lo = j1; index_lo < j2; ++index_lo) {
        const int64_t index_hi = index_lo + t;
        const uint64_t lo_val = out[index_lo];
        const uint64_t hi_val = out[index_hi];
        uint64_t omega_factor = lo_val;
        if (omega_factor < hi_val) {
          omega_factor += modulus;
        }
        omega_factor -= hi_val;
        out[index_lo] = mod_add(lo_val, hi_val, modulus);
        out[index_hi] = mod_mul(omega_factor, omega, modulus);
      }
    }
    if (m == 1) {
      break;
    }
    t <<= 1;
    ++logt1;
    m >>= 1;
  }

  const uint64_t ring_dim_inv = mod_pow(static_cast<uint64_t>(ring_dim), modulus - 2, modulus);
  for (int64_t i = 0; i < ring_dim; ++i) {
    out[i] = mod_mul(out[i], ring_dim_inv, modulus);
  }
}

std::vector<std::vector<uint64_t>> garner_inverses(const uint64_t* moduli, int64_t limbs) {
  std::vector<std::vector<uint64_t>> inverses(
      static_cast<size_t>(limbs), std::vector<uint64_t>(static_cast<size_t>(limbs), 0));
  for (int64_t i = 0; i < limbs; ++i) {
    for (int64_t j = 0; j < i; ++j) {
      inverses[i][j] = mod_pow(moduli[j] % moduli[i], moduli[i] - 2, moduli[i]);
    }
  }
  return inverses;
}

long double mixed_radix_fraction(
    const std::vector<uint64_t>& digits,
    const uint64_t* moduli,
    int64_t limbs) {
  long double value = 0.0L;
  for (int64_t i = limbs - 1; i >= 0; --i) {
    value = (value + static_cast<long double>(digits[static_cast<size_t>(i)])) /
        static_cast<long double>(moduli[i]);
  }
  return value;
}

double mixed_radix_to_double(
    const std::vector<uint64_t>& digits,
    const uint64_t* moduli,
    int64_t limbs) {
  long double value = 0.0L;
  for (int64_t i = limbs - 1; i >= 0; --i) {
    value = value * static_cast<long double>(moduli[i]) +
        static_cast<long double>(digits[static_cast<size_t>(i)]);
  }
  return static_cast<double>(value);
}

std::vector<uint64_t> mixed_radix_complement(
    const std::vector<uint64_t>& digits,
    const uint64_t* moduli,
    int64_t limbs) {
  std::vector<uint64_t> result(static_cast<size_t>(limbs), 0);
  uint64_t borrow = 0;
  for (int64_t i = 0; i < limbs; ++i) {
    const uint64_t digit = digits[static_cast<size_t>(i)];
    const uint64_t modulus = moduli[i];
    if (digit == 0 && borrow == 0) {
      result[static_cast<size_t>(i)] = 0;
      borrow = 0;
    } else {
      result[static_cast<size_t>(i)] = modulus - digit - borrow;
      borrow = 1;
    }
  }
  return result;
}

double centered_crt_to_double(
    const uint64_t* coeff_limbs,
    int64_t coeff_idx,
    int64_t ring_dim,
    const uint64_t* moduli,
    int64_t limbs,
    const std::vector<std::vector<uint64_t>>& inverses) {
  std::vector<uint64_t> digits(static_cast<size_t>(limbs), 0);
  for (int64_t i = 0; i < limbs; ++i) {
    uint64_t value = coeff_limbs[i * ring_dim + coeff_idx] % moduli[i];
    for (int64_t j = 0; j < i; ++j) {
      const uint64_t digit = digits[static_cast<size_t>(j)] % moduli[i];
      value = (value >= digit) ? (value - digit) : (value + moduli[i] - digit);
      value = mod_mul(value, inverses[i][j], moduli[i]);
    }
    digits[static_cast<size_t>(i)] = value;
  }

  if (mixed_radix_fraction(digits, moduli, limbs) <= 0.5L) {
    return mixed_radix_to_double(digits, moduli, limbs);
  }
  return -mixed_radix_to_double(mixed_radix_complement(digits, moduli, limbs), moduli, limbs);
}

void bit_reverse_complex(std::vector<std::complex<double>>& values) {
  const size_t size = values.size();
  for (size_t i = 1, j = 0; i < size; ++i) {
    size_t bit = size >> 1;
    while (j >= bit) {
      j -= bit;
      bit >>= 1;
    }
    j += bit;
    if (i < j) {
      std::swap(values[i], values[j]);
    }
  }
}

void fft_special(std::vector<std::complex<double>>& values, int64_t cycl_order) {
  const size_t vals_size = values.size();
  std::vector<int64_t> rot_group(vals_size);
  int64_t five_pows = 1;
  for (size_t i = 0; i < vals_size; ++i) {
    rot_group[i] = five_pows;
    five_pows = (five_pows * 5) % cycl_order;
  }

  std::vector<std::complex<double>> ksi_pows(static_cast<size_t>(cycl_order) + 1);
  for (int64_t j = 0; j < cycl_order; ++j) {
    const double angle = 2.0 * kPi * static_cast<double>(j) / static_cast<double>(cycl_order);
    ksi_pows[static_cast<size_t>(j)] = std::complex<double>(std::cos(angle), std::sin(angle));
  }
  ksi_pows[static_cast<size_t>(cycl_order)] = ksi_pows[0];

  bit_reverse_complex(values);
  for (size_t length = 2; length <= vals_size; length <<= 1) {
    const size_t lenh = length >> 1;
    const size_t lenq = length << 2;
    const size_t gap = static_cast<size_t>(cycl_order) / lenq;
    for (size_t i = 0; i < vals_size; i += length) {
      for (size_t j = 0; j < lenh; ++j) {
        const size_t idx = static_cast<size_t>(rot_group[j] % static_cast<int64_t>(lenq)) * gap;
        const std::complex<double> u = values[i + j];
        const std::complex<double> v = values[i + j + lenh] * ksi_pows[idx];
        values[i + j] = u + v;
        values[i + j + lenh] = u - v;
      }
    }
  }
}

std::vector<std::complex<double>> conjugate_slots(
    const std::vector<std::complex<double>>& values) {
  std::vector<std::complex<double>> result(values.size(), std::complex<double>(0.0, 0.0));
  if (!values.empty()) {
    result[0] = std::complex<double>(values[0].real(), -values[0].imag());
  }
  for (size_t i = 1; i < values.size(); ++i) {
    result[i] = std::complex<double>(-values[values.size() - i].imag(), -values[values.size() - i].real());
  }
  return result;
}

} // namespace

namespace at::native {

Tensor ckks_decrypt_decode_cpu(
    const Tensor& ct0,
    const Tensor& ct1,
    const Tensor& secret_key,
    const Tensor& moduli_q,
    const Tensor& roots_q,
    int64_t cur_limbs,
    int64_t plaintext_modulus_bits,
    int64_t noise_scale_deg,
    int64_t slots) {
  TORCH_CHECK(ct0.scalar_type() == at::kUInt64, "ct0 must be uint64");
  TORCH_CHECK(ct1.scalar_type() == at::kUInt64, "ct1 must be uint64");
  TORCH_CHECK(secret_key.scalar_type() == at::kUInt64, "secret_key must be uint64");
  TORCH_CHECK(moduli_q.scalar_type() == at::kUInt64, "moduli_q must be uint64");
  TORCH_CHECK(roots_q.scalar_type() == at::kUInt64, "roots_q must be uint64");
  TORCH_CHECK(ct0.dim() == 2, "ct0 must be [limbs, N]");
  TORCH_CHECK(ct1.dim() == 2, "ct1 must be [limbs, N]");
  TORCH_CHECK(secret_key.dim() == 2, "secret_key must be [limbs, N]");
  TORCH_CHECK(ct0.sizes() == ct1.sizes(), "ct0/ct1 shape mismatch");
  TORCH_CHECK(cur_limbs > 0 && cur_limbs <= ct0.size(0), "invalid cur_limbs");
  TORCH_CHECK(secret_key.size(0) >= cur_limbs && secret_key.size(1) == ct0.size(1), "secret_key shape mismatch");
  TORCH_CHECK(moduli_q.numel() >= cur_limbs, "not enough Q moduli");
  TORCH_CHECK(roots_q.numel() >= cur_limbs, "not enough Q roots");

  const int64_t ring_dim = ct0.size(1);
  TORCH_CHECK(ring_dim > 0 && (ring_dim & (ring_dim - 1)) == 0, "ring_dim must be a power of two");
  const int64_t nh = ring_dim >> 1;
  slots = slots == 0 ? nh : slots;
  TORCH_CHECK(slots > 0 && slots <= nh && nh % slots == 0, "invalid CKKS slots");
  const int64_t gap = nh / slots;

  Tensor ct0_contig = ct0.contiguous();
  Tensor ct1_contig = ct1.contiguous();
  Tensor sk_contig = secret_key.contiguous();
  Tensor moduli_contig = moduli_q.contiguous();
  Tensor roots_contig = roots_q.contiguous();

  const uint64_t* ct0_ptr = ct0_contig.data_ptr<uint64_t>();
  const uint64_t* ct1_ptr = ct1_contig.data_ptr<uint64_t>();
  const uint64_t* sk_ptr = sk_contig.data_ptr<uint64_t>();
  const uint64_t* moduli_ptr = moduli_contig.data_ptr<uint64_t>();
  const uint64_t* roots_ptr = roots_contig.data_ptr<uint64_t>();

  Tensor phase = at::empty({cur_limbs, ring_dim}, ct0.options().device(at::kCPU));
  Tensor coeffs = at::empty_like(phase);
  uint64_t* phase_ptr = phase.data_ptr<uint64_t>();
  uint64_t* coeffs_ptr = coeffs.data_ptr<uint64_t>();

  const int max_threads = omp_get_max_threads();
#pragma omp parallel for collapse(2) schedule(static) num_threads(max_threads)
  for (int64_t limb = 0; limb < cur_limbs; ++limb) {
    for (int64_t idx = 0; idx < ring_dim; ++idx) {
      const uint64_t modulus = moduli_ptr[limb];
      phase_ptr[limb * ring_dim + idx] = mod_add(
          mod_mul(ct1_ptr[limb * ring_dim + idx], sk_ptr[limb * ring_dim + idx], modulus),
          ct0_ptr[limb * ring_dim + idx],
          modulus);
    }
  }

#pragma omp parallel for schedule(static) num_threads(max_threads)
  for (int64_t limb = 0; limb < cur_limbs; ++limb) {
    inverse_ntt_from_eval(
        phase_ptr + limb * ring_dim,
        coeffs_ptr + limb * ring_dim,
        ring_dim,
        roots_ptr[limb],
        moduli_ptr[limb]);
  }

  const auto inverses = garner_inverses(moduli_ptr, cur_limbs);
  const double scaling_pre = std::pow(
      2.0,
      -static_cast<double>(plaintext_modulus_bits) *
          static_cast<double>(std::max<int64_t>(noise_scale_deg - 1, 0)));
  std::vector<std::complex<double>> cur_values(static_cast<size_t>(slots));
  for (int64_t slot = 0; slot < slots; ++slot) {
    const int64_t idx = slot * gap;
    const double real = centered_crt_to_double(coeffs_ptr, idx, ring_dim, moduli_ptr, cur_limbs, inverses);
    const double imag = centered_crt_to_double(coeffs_ptr, idx + nh, ring_dim, moduli_ptr, cur_limbs, inverses);
    cur_values[static_cast<size_t>(slot)] =
        std::complex<double>(real * scaling_pre, imag * scaling_pre);
  }

  const std::vector<std::complex<double>> conjugate = conjugate_slots(cur_values);
  const double scale = 0.5 * std::pow(2.0, -static_cast<double>(plaintext_modulus_bits));
  std::vector<std::complex<double>> real_values(static_cast<size_t>(slots));
  for (int64_t slot = 0; slot < slots; ++slot) {
    real_values[static_cast<size_t>(slot)] = std::complex<double>(
        scale * (cur_values[static_cast<size_t>(slot)].real() + conjugate[static_cast<size_t>(slot)].real()),
        scale * (cur_values[static_cast<size_t>(slot)].imag() + conjugate[static_cast<size_t>(slot)].imag()));
  }
  fft_special(real_values, ring_dim * 2);

  Tensor output = at::empty({slots}, at::TensorOptions().dtype(at::kDouble).device(at::kCPU));
  double* output_ptr = output.data_ptr<double>();
  for (int64_t slot = 0; slot < slots; ++slot) {
    output_ptr[slot] = real_values[static_cast<size_t>(slot)].real();
  }
  return output;
}

} // namespace at::native
