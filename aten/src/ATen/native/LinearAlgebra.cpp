#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/OpMathType.h>
#include <ATen/Parallel.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <ATen/TensorOperators.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/CPUBlas.h>
#include <ATen/native/LinearAlgebra.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/mkldnn/Matmul.h>
#include <ATen/native/mkldnn/Utils.h>
#include <c10/core/GradMode.h>
#include <c10/util/irange.h>
#include <tuple>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_addmm_activation_native.h>
#include <ATen/ops/addmm_native.h>
#include <ATen/ops/addbmm_native.h>
#include <ATen/ops/baddbmm_native.h>
#include <ATen/ops/bmm.h>
#include <ATen/ops/bmm_native.h>
#include <ATen/ops/dot.h>
#include <ATen/ops/dot_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/matmul.h>
#include <ATen/ops/matmul_native.h>
#include <ATen/ops/matmul_backward_native.h>
#include <ATen/ops/mm.h>
#include <ATen/ops/mm_native.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/mv.h>
#include <ATen/ops/outer.h>
#include <ATen/ops/outer_native.h>
#include <ATen/ops/ger_native.h>
#include <ATen/ops/_unsafe_view.h>
#include <ATen/ops/vdot_native.h>
#endif

namespace at::meta {

#define ADDMM_META() \
  TORCH_CHECK(self.scalar_type() == mat2.scalar_type(), "self and mat2 must have the same dtype, but got ", self.scalar_type(), " and ", mat2.scalar_type()); \
  TORCH_CHECK(mat1.scalar_type() == mat2.scalar_type(), "mat1 and mat2 must have the same dtype, but got ", mat1.scalar_type(), " and ", mat2.scalar_type()); \
  TORCH_CHECK(mat1.dim() == 2, "mat1 must be a matrix, got ", mat1.dim(), "-D tensor"); \
  TORCH_CHECK(mat2.dim() == 2, "mat2 must be a matrix, got ", mat2.dim(), "-D tensor"); \
  TORCH_CHECK( \
      mat1.sizes()[1] == mat2.sizes()[0], "mat1 and mat2 shapes cannot be multiplied (", \
      mat1.sizes()[0], "x", mat1.sizes()[1], " and ", mat2.sizes()[0], "x", mat2.sizes()[1], ")"); \
 \
  auto names = at::namedinference::propagate_names_for_addmm(mat1, mat2, self); \
  set_output_raw_strided(0, {mat1.sizes()[0], mat2.sizes()[1]}, {}, mat1.options(), names);

TORCH_META_FUNC(addmm)(const Tensor& self, const Tensor& mat1, const Tensor& mat2, const Scalar& beta, const Scalar& alpha) {
  ADDMM_META();
}

TORCH_META_FUNC(_addmm_activation)(const Tensor& self, const Tensor& mat1, const Tensor& mat2, const Scalar& beta, const Scalar& alpha, bool use_gelu) {
  ADDMM_META();
}

TORCH_META_FUNC(mm)(const Tensor & self, const Tensor & mat2) {
  TORCH_CHECK(self.dim() == 2, "self must be a matrix");
  TORCH_CHECK(mat2.dim() == 2, "mat2 must be a matrix");
  TORCH_CHECK(
      self.sizes()[1] == mat2.sizes()[0], "mat1 and mat2 shapes cannot be multiplied (",
      self.sizes()[0], "x", self.sizes()[1], " and ", mat2.sizes()[0], "x", mat2.sizes()[1], ")");

  auto names = at::namedinference::compute_matmul_outnames(self, mat2);
  set_output_raw_strided(0, {self.sizes()[0], mat2.sizes()[1]}, {}, self.options(), names);
}

template <typename Meta>
static void common_checks_baddbmm_bmm(Meta& meta, const Tensor& batch1, const Tensor& batch2, const Scalar& beta, const Scalar& alpha, bool is_bmm, const std::optional<Tensor>& self_baddbmm = std::nullopt) {
  TORCH_CHECK(batch1.dim() == 3, "batch1 must be a 3D tensor");
  TORCH_CHECK(batch2.dim() == 3, "batch2 must be a 3D tensor");

  const auto batch1_sizes = batch1.sizes();
  const auto batch2_sizes = batch2.sizes();

  int64_t bs = batch1_sizes[0];
  int64_t contraction_size = batch1_sizes[2];
  int64_t res_rows = batch1_sizes[1];
  int64_t res_cols = batch2_sizes[2];
  std::vector<int64_t> output_size {bs, res_rows, res_cols};

  TORCH_CHECK(batch2_sizes[0] == bs && batch2_sizes[1] == contraction_size,
              "Expected size for first two dimensions of batch2 tensor to be: [",
              bs, ", ", contraction_size, "] but got: [", batch2_sizes[0], ", ", batch2_sizes[1], "].");

  auto& result = meta.maybe_get_output(0);
  meta.set_output_raw_strided(0, output_size, {}, batch2.options());
  const auto result_sizes = result.sizes();
  TORCH_CHECK(result_sizes == output_size,
              "Expected an output tensor with shape [", output_size, "] but got shape ", result_sizes);

  std::vector<Dimname> outnames = {};
  if (!is_bmm) {
    if (self_baddbmm.has_value()) {
      const auto& self = self_baddbmm.value();
      if (beta.toComplexDouble() != 0.0) result.copy_(self);
      TORCH_CHECK(self.dim() == 3, "self must be a 3D tensor");
      const auto self_sizes = self.sizes();
      TORCH_CHECK(self_sizes == output_size,
                  "Expected an input tensor shape with shape ", output_size, " but got shape: ", self_sizes);
      outnames = namedinference::compute_baddbmm_outnames(result, batch1, batch2, self);
    }
  } else {
    outnames = namedinference::compute_bmm_outnames(result, batch1, batch2);
  }

  namedinference::propagate_names_if_nonempty(result, outnames);
}

TORCH_META_FUNC(bmm)(const Tensor& self, const Tensor& mat2) {
    common_checks_baddbmm_bmm(*this, self, mat2, Scalar(0.0), Scalar(1.0), true);
}

TORCH_META_FUNC(baddbmm)(const Tensor& self, const Tensor& batch1, const Tensor& batch2, const Scalar& beta, const Scalar& alpha) {
  auto self_ = expand_size(self, {batch1.size(0), batch1.size(1), batch2.size(2)}, "baddbmm");
  TORCH_CHECK(self.dtype() == batch1.dtype(), "Input dtypes must be the same, got: input ", self.dtype(), ", batch1: ", batch1.dtype(), ", batch2: ", batch2.dtype());
  common_checks_baddbmm_bmm(*this, batch1, batch2, beta, alpha, false, *self_);
}

} // namespace at::meta

namespace at::native {

DEFINE_DISPATCH(addr_stub);

#if !defined(C10_MOBILE)
#define _AT_DISPATCH_ADDMM_TYPES(TYPE, NAME, ...)                                               \
        AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND6(                                                 \
            kBFloat16, kHalf, kFloat8_e5m2, kFloat8_e4m3fn, kFloat8_e5m2fnuz, kFloat8_e4m3fnuz, \
            TYPE, NAME, __VA_ARGS__)
#else
#define _AT_DISPATCH_ADDMM_TYPES(TYPE, NAME, ...)        \
        AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(kBFloat16, \
            TYPE, NAME, __VA_ARGS__)
#endif

static void check_1d(const Tensor& t, const char* arg, const char* fn) {
 TORCH_CHECK(t.dim() == 1, fn, ": Expected 1-D argument ", arg, ", but got ", t.dim(), "-D");
}

Tensor& outer_out(const Tensor& self, const Tensor& vec2, Tensor &result) {
  check_1d(self, "self", "outer");
  check_1d(vec2, "vec2", "outer");
  at::mul_out(result, self.reshape({self.size(0), 1}), vec2);
  return result;
}

Tensor outer(const Tensor& self, const Tensor& vec2) {
  check_1d(self, "self", "outer");
  check_1d(vec2, "vec2", "outer");
  return self.reshape_symint({self.sym_size(0), 1}) * vec2;
}

Tensor& ger_out(const Tensor& self, const Tensor& vec2, Tensor &result) {
  return at::outer_out(result, self, vec2);
}

Tensor ger(const Tensor& self, const Tensor& vec2) {
  return self.outer(vec2);
}

static void addmm_impl_cpu_(
    Tensor &result, const Tensor &self, Tensor m1, Tensor m2, const Scalar& beta, const Scalar& alpha) {
  TORCH_INTERNAL_ASSERT(self.dim() == 2 && m1.dim() == 2 && m2.dim() == 2);

  TORCH_CHECK(
    m1.dtype() == m2.dtype(),
    "expected m1 and m2 to have the same dtype, but got: ", m1.dtype(), " != ", m2.dtype()
  )

  const auto self_sizes = self.sizes();
  auto m1_strides = m1.strides();
  auto m1_sizes = m1.sizes();
  auto m2_strides = m2.strides();
  auto m2_sizes = m2.sizes();

  TORCH_CHECK(
      self_sizes[0] == m1_sizes[0] && self_sizes[1] == m2_sizes[1],
      "input shape is incompatible with matrix multiplication (",
      m1_sizes[0], "x", m1_sizes[1], " @ ", m2_sizes[0], "x", m2_sizes[1], " != ",
      self_sizes[0], "x", self_sizes[1], ")");

  at::native::resize_output(result, self_sizes);
  const auto result_strides = result.strides();
  const auto result_sizes = result.sizes();

  if (result.numel() == 0) {
    return;
  }

  if (m1_sizes[1] == 0) {
    if (beta.toComplexDouble() == 0.0) {
      result.zero_();
    } else {
      if (!self.is_same(result)) {
        result.copy_(self);
      }
      result.mul_(beta);
    }
    return;
  }

  if (beta.toComplexDouble() != 0.0 && !self.is_same(result)) {
    result.copy_(self);
  }

  bool transpose_c = false;
  Tensor c;

  if (result_strides[0] == 1 &&
      (result_sizes[1] == 1 || result_strides[1] >= std::max(int64_t{1}, result_sizes[0]))) {
    transpose_c = false;
    c = result.resolve_conj();
  } else if (result_strides[1] == 1 &&
             (result_sizes[0] == 1 || result_strides[0] >= std::max(int64_t{1}, result_sizes[1]))) {
    std::swap(m1, m2);
    std::swap(m1_sizes, m2_sizes);
    std::swap(m1_strides, m2_strides);
    transpose_c = true;
    c = result.resolve_conj();
  } else {
    transpose_c = false;
    c = result.resolve_conj().transpose(0, 1).contiguous().transpose_(0, 1);
  }

  const int64_t m = result_sizes[transpose_c ? 1 : 0];
  const int64_t n = result_sizes[transpose_c ? 0 : 1];
  const int64_t k = m1_sizes[transpose_c ? 0 : 1];

  bool transpose_a = false;
  Tensor a;
  if (m1_strides[transpose_c ? 1 : 0] == 1 &&
      m1_strides[transpose_c ? 0 : 1] >= std::max(int64_t{1}, m)) {
    transpose_a = false;
    a = m1.resolve_conj();
  } else if (m1_strides[transpose_c ? 0 : 1] == 1 &&
             m1_strides[transpose_c ? 1 : 0] >= std::max(int64_t{1}, k)) {
    transpose_a = true;
    a = m1;
  } else {
    transpose_a = !transpose_c;
    a = m1.clone(at::MemoryFormat::Contiguous);
  }

  bool transpose_b = false;
  Tensor b;
  if (m2_strides[transpose_c ? 1 : 0] == 1 &&
      m2_strides[transpose_c ? 0 : 1] >= std::max(int64_t{1}, k)) {
    transpose_b = false;
    b = m2.resolve_conj();
  } else if (m2_strides[transpose_c ? 0 : 1] == 1 &&
             m2_strides[transpose_c ? 1 : 0] >= std::max(int64_t{1}, n)) {
    transpose_b = true;
    b = m2;
  } else {
    transpose_b = !transpose_c;
    b = m2.clone(at::MemoryFormat::Contiguous);
  }

  const int64_t lda = a.strides()[(transpose_a == transpose_c) ? 1 : 0];
  const int64_t ldb = b.strides()[(transpose_b == transpose_c) ? 1 : 0];
  const int64_t ldc = c.strides()[transpose_c ? 0 : 1];

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!c.is_conj());

  _AT_DISPATCH_ADDMM_TYPES(result.scalar_type(), "addmm_impl_cpu_", [&]{
        using opmath_t = at::opmath_type<scalar_t>;
        at::native::cpublas::gemm(
            transpose_a ? a.is_conj() ? TransposeType::ConjTranspose : TransposeType::Transpose : TransposeType::NoTranspose,
            transpose_b ? b.is_conj() ? TransposeType::ConjTranspose : TransposeType::Transpose : TransposeType::NoTranspose,
            m, n, k,
            alpha.to<opmath_t>(),
            a.const_data_ptr<scalar_t>(), lda,
            b.const_data_ptr<scalar_t>(), ldb,
            beta.to<opmath_t>(),
            c.mutable_data_ptr<scalar_t>(), ldc);
      });

  if (!c.is_same(result)) {
    result.copy_(c);
  }
}

static void addbmm_impl_(
    Tensor &result, const Tensor &self, const Tensor &batch1, const Tensor &batch2, const Scalar& beta, const Scalar& alpha) {
  TORCH_CHECK(batch1.dim() == 3, "batch1 must be a 3D tensor");
  TORCH_CHECK(batch2.dim() == 3, "batch2 must be a 3D tensor");
  TORCH_CHECK(batch1.size(0) == batch2.size(0),
      "batch1 and batch2 must have same number of batches, got ",
      batch1.size(0), " and ", batch2.size(0));
  TORCH_CHECK(batch1.size(2) == batch2.size(1),
      "Incompatible matrix sizes for bmm (",
      batch1.size(1), "x", batch1.size(2), " and ",
      batch2.size(1), "x", batch2.size(2), ")");

  const int64_t dim1 = batch1.size(1);
  const int64_t dim2 = batch2.size(2);
  TORCH_CHECK(self.size(0) == dim1 && self.size(1) == dim2,
      "self tensor does not match matmul output shape");

  result.resize_as_(self);

  if (beta.to<c10::complex<double>>() != 0.0 && !self.is_same(result)) {
    result.copy_(self);
  }

  const int64_t num_batches = batch1.size(0);

  if (num_batches == 0) {
    if (beta.to<c10::complex<double>>() != 0.0) {
      result.mul_(beta);
    } else {
      result.zero_();
    }
    return;
  }

  auto adjusted_beta(beta);
  for (const auto batch : c10::irange(num_batches)) {
    result.addmm_(batch1[batch], batch2[batch], adjusted_beta, alpha);
    adjusted_beta = 1;
  }
}

Tensor& addbmm_out(const Tensor& self, const Tensor& batch1, const Tensor& batch2, const Scalar& beta, const Scalar& alpha, Tensor& result) {
  auto b_self = expand_size(self, {batch1.size(1), batch2.size(2)}, "addbmm_out");
  {
    at::NoNamesGuard guard;
    addbmm_impl_(result, *b_self, batch1, batch2, beta, alpha);
  }
  auto names = at::namedinference::propagate_names_for_addmm(batch1, batch2, self);
  at::namedinference::propagate_names_if_nonempty(result, names);
  return result;
}

Tensor &addbmm_(Tensor& self, const Tensor& batch1, const Tensor& batch2, const Scalar& beta, const Scalar& alpha) {
  return native::addbmm_out(self, batch1, batch2, beta, alpha, self);
}

Tensor addbmm(const Tensor& self, const Tensor& batch1, const Tensor& batch2, const Scalar& beta, const Scalar& alpha) {
  Tensor result = at::empty({0}, self.options());
  return native::addbmm_out(self, batch1, batch2, beta, alpha, result);
}

TORCH_IMPL_FUNC(addmm_out_cpu)(const Tensor& self, const Tensor& mat1, const Tensor& mat2, const Scalar& beta, const Scalar& alpha, const Tensor &result) {
  auto b_self = expand_size(self, {mat1.sizes()[0], mat2.sizes()[1]}, "addmm_out");
  {
    at::NoNamesGuard guard;
    addmm_impl_cpu_(const_cast<Tensor&>(result), *b_self, mat1, mat2, beta, alpha);
  }
}

TORCH_IMPL_FUNC(addmm_activation_out_cpu)(const Tensor& self, const Tensor& mat1, const Tensor& mat2, const Scalar& beta, const Scalar& alpha, bool use_gelu, const Tensor &result) {
  auto b_self = expand_size(self, {mat1.sizes()[0], mat2.sizes()[1]}, "addmm_out");
  {
    at::NoNamesGuard guard;
    addmm_impl_cpu_(const_cast<Tensor&>(result), *b_self, mat1, mat2, beta, alpha);
  }
}

TORCH_IMPL_FUNC(mm_out_cpu)(const Tensor & self, const Tensor & mat2, const Tensor & result) {
  {
    at::NoNamesGuard guard;
    addmm_impl_cpu_(const_cast<Tensor&>(result), result, self, mat2, 0, 1);
  }
}

template <typename scalar_t, bool is_bmm>
static inline void baddbmm_cpu_kernel(const Tensor& result, const Tensor& self, const Tensor& mat2, const Scalar& beta_, const Scalar& alpha_) {
  int64_t bs = result.size(0);
  int64_t is = result.size(1);
  int64_t js = result.size(2);
  int64_t ks = self.size(2);

  using opmath_t = at::opmath_type<scalar_t>;
  opmath_t alpha = alpha_.to<opmath_t>();
  opmath_t beta = beta_.to<opmath_t>();

  auto r0 = result.accessor<scalar_t, 3>();
  auto s0 = self.accessor<const scalar_t, 3>();
  auto m0 = mat2.accessor<const scalar_t, 3>();

  int64_t grain_size = std::max(internal::GRAIN_SIZE / (is * js * ks), static_cast<int64_t>(1));
  parallel_for(0, bs, grain_size, [&](int64_t b_begin, int64_t b_end) {
      for (const auto b : c10::irange(b_begin, b_end)) {
        auto r1 = r0[b];
        auto s1 = s0[b];
        auto m1 = m0[b];
        for (const auto i : c10::irange(is)) {
          auto r2 = r1[i];
          auto s2 = s1[i];
          for (const auto j : c10::irange(js)) {
            opmath_t acc_value = 0;
            for (const auto k : c10::irange(ks)) {
              acc_value += static_cast<opmath_t>(s2[k]) *
                  static_cast<opmath_t>(m1[k][j]);
            }
            if (is_bmm) {
              r2[j] = acc_value;
            } else {
              if (beta == opmath_t{0}) {
                r2[j] = alpha * acc_value;
              } else {
                r2[j] = static_cast<opmath_t>(r2[j]) * beta + alpha * acc_value;
              }
            }
          }
        }
      }
    });
}

static void baddbmm_with_gemm_(const Tensor &result, const Tensor &mat1, const Tensor &mat2, const Scalar &beta_, const Scalar &alpha_) {
  TORCH_INTERNAL_ASSERT(result.is_contiguous());

  const auto result_sizes = result.sizes();
  const auto result_strides = result.strides();
  const auto mat1_strides = mat1.strides();
  const auto mat2_strides = mat2.strides();
  const auto mat1_sizes = mat1.sizes();
  const auto mat2_sizes = mat2.sizes();

  auto is_transposed = [](const c10::IntArrayRef& strides, const c10::IntArrayRef& sizes) {
    return strides[1] == 1 && strides[2] >= sizes[1];
  };

  const auto transpose_a = is_transposed(mat2_strides, mat2_sizes);
  const auto transpose_b = is_transposed(mat1_strides, mat1_sizes);

  const int64_t batch_size = mat1_sizes[0];
  const int64_t m = result_sizes[2];
  const int64_t n = result_sizes[1];
  const int64_t k = mat2_sizes[1];

  const int64_t lda = mat2_strides[transpose_a ? 2 : 1];
  const int64_t ldb = mat1_strides[transpose_b ? 2 : 1];
  const int64_t ldc = result_strides[1];

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(result.scalar_type(), "baddbmm_with_gemm", [&] {
    using opmath_t = at::opmath_type<scalar_t>;
    const auto alpha = alpha_.to<opmath_t>();
    const auto beta = beta_.to<opmath_t>();
    at::native::cpublas::gemm_batched_with_stride(
        transpose_a ? TransposeType::Transpose : TransposeType::NoTranspose,
        transpose_b ? TransposeType::Transpose : TransposeType::NoTranspose,
        batch_size, m, n, k, alpha,
        mat2.const_data_ptr<scalar_t>(), lda, mat2_strides[0],
        mat1.const_data_ptr<scalar_t>(), ldb, mat1_strides[0],
        beta,
        result.data_ptr<scalar_t>(), ldc, result_strides[0]);
  });
}

static inline void bmm_out_or_baddbmm_(const Tensor& self_or_result_, const Tensor& batch1, const Tensor& batch2, const Scalar& beta, const Scalar& alpha, bool is_bmm_out) {
  Tensor& self_or_result = const_cast<Tensor&>(self_or_result_);

  const auto batch1_sizes = batch1.sizes();
  const auto batch2_sizes = batch2.sizes();

  int64_t bs = batch1_sizes[0];
  int64_t contraction_size = batch1_sizes[2];
  int64_t res_rows = batch1_sizes[1];
  int64_t res_cols = batch2_sizes[2];

  if (self_or_result.numel() == 0) {
    return;
  } else if (contraction_size == 0) {
    if (is_bmm_out || (beta.to<c10::complex<double>>() == 0.0)) {
      self_or_result.zero_();
      return;
    } else {
      self_or_result.mul_(beta);
      return;
    }
  }

  auto batch_items_contiguous_or_transposed = [&](const Tensor& t) {
    const auto sizes = t.sizes();
    const auto strides = t.strides();
    return (strides[2] == 1 && (sizes[1] == 1 || strides[1] >= sizes[2])) ||
        (strides[1] == 1 && (sizes[2] == 1 || strides[2] >= sizes[1]));
  };

  if (contraction_size * res_rows * res_cols < 400) {
    if (is_bmm_out) {
      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kBFloat16, kHalf, batch1.scalar_type(), "bmm", [&] {
          baddbmm_cpu_kernel<scalar_t, true>(self_or_result, batch1, batch2, beta, alpha);
        });
    } else {
      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kBFloat16, kHalf, batch1.scalar_type(), "baddbmm", [&] {
          baddbmm_cpu_kernel<scalar_t, false>(self_or_result, batch1, batch2, beta, alpha);
        });
    }
  } else if (at::hasMKL() && ((
            self_or_result.scalar_type() != kBFloat16 &&
            self_or_result.scalar_type() != kHalf &&
            self_or_result.is_floating_point()) ||
            self_or_result.is_complex())
            && batch_items_contiguous_or_transposed(batch1)
            && batch_items_contiguous_or_transposed(batch2)
            && self_or_result.is_contiguous()) {
    baddbmm_with_gemm_(self_or_result, batch1, batch2, beta, alpha);
  } else {
    if (is_bmm_out) {
      for (const auto b : c10::irange(bs)) {
        auto r = self_or_result.select(0, b);
        addmm_impl_cpu_(r, r, batch1.select(0, b), batch2.select(0, b), 0, 1);
      }
    } else {
      for (const auto b : c10::irange(bs)) {
        self_or_result.select(0, b).addmm_(
            batch1.select(0, b), batch2.select(0, b), beta, alpha);
      }
    }
  }
}

static void conjugate_mutable_input_if_needed(const Tensor& self, bool conjugate) {
  if (conjugate) {
    self.conj_physical_();
  }
}

TORCH_IMPL_FUNC(baddbmm_out_cpu)
(const Tensor & self, const Tensor & batch1, const Tensor & batch2, const Scalar& beta, const Scalar& alpha, const Tensor& result) {
    bool self_is_conj = result.is_conj();
    conjugate_mutable_input_if_needed(result, self_is_conj);
    bmm_out_or_baddbmm_(result, batch1.resolve_conj(), batch2.resolve_conj(), beta, alpha, false);
    conjugate_mutable_input_if_needed(result, self_is_conj);
}

TORCH_IMPL_FUNC(bmm_out_cpu)
(const Tensor & batch1, const Tensor & batch2, const Tensor & result) {
    {
    NoNamesGuard guard;
    bool result_is_conj = result.is_conj();
    conjugate_mutable_input_if_needed(result, result_is_conj);
    bmm_out_or_baddbmm_(result, batch1.resolve_conj(), batch2.resolve_conj(), Scalar(0.0), Scalar(1.0), true);
    conjugate_mutable_input_if_needed(result, result_is_conj);
    }
}

Tensor& dot_out(const Tensor& self, const Tensor& other, Tensor& result) {
  auto output_device = result.device();
  auto input1_device = self.device();
  auto input2_device = other.device();
  TORCH_CHECK(
    (output_device == input1_device) && (input1_device == input2_device),
    "dot: Expected the output and input tensors to be on the "
    "same device, but got the output tensor on ", output_device,
    ", the 'input' tensor on ", input1_device, ", and the 'other' tensor on ", input2_device);
  at::native::resize_output(result, {});
  TORCH_CHECK(result.scalar_type() == self.scalar_type(),
           "result dtype ", result.scalar_type(), " does not match input dtype ", self.scalar_type());
  return result.fill_(self.dot(other));
}

Tensor& vdot_out(const Tensor& self, const Tensor& other, Tensor& result) {
  auto output_device = result.device();
  auto input1_device = self.device();
  auto input2_device = other.device();
  TORCH_CHECK(
    (output_device == input1_device) && (input1_device == input2_device),
    "vdot: Expected the output and input tensors to be on the "
    "same device, but got the output tensor on ", output_device,
    ", the 'input' tensor on ", input1_device, ", and the 'other' tensor on ", input2_device);
  at::native::resize_output(result, {});
  TORCH_CHECK(result.scalar_type() == self.scalar_type(),
           "result dtype ", result.scalar_type(), " does not match input dtype ", self.scalar_type());
  return result.fill_(self.vdot(other));
}

static bool should_fold(const Tensor& tensor1, const Tensor& tensor2, bool has_out) {
  const auto tensor1_larger = tensor1.dim() >= tensor2.dim();

  const auto t1 = tensor1_larger ? MaybeOwned<Tensor>::borrowed(tensor1)
                                 : MaybeOwned<Tensor>::owned(tensor2.mT());
  const int64_t dim_t1 = t1->dim();
  const auto dim_t2 = tensor1_larger ? tensor2.dim()
                                     : tensor1.dim();

  if (!(dim_t1 >= 3 && dim_t2 <= 2)) {
    return false;
  }

  bool t2_requires_grad = tensor1_larger ? tensor2.requires_grad() : tensor1.requires_grad();
  if (t2_requires_grad && !has_out) {
    return true;
  }

  if (tensor1.dim() == 2) {
    return false;
  }

  if (t1->numel() == 0) {
    return true;
  }

  const auto t1_shape = t1->sizes();
  const auto t1_strides = t1->strides();
  for (auto i = int64_t{0}; i < dim_t1 - int64_t{2}; ++i) {
    if (t1_strides[i] != t1_strides[i+1] * t1_shape[i+1]) {
      return false;
    }
  }
  return true;
}

static Tensor _matmul_impl(
    Tensor& out,
    const Tensor& tensor1,
    const Tensor& tensor2) {
  NoNamesGuard guard;
  const auto dim_tensor1 = tensor1.dim();
  const auto dim_tensor2 = tensor2.dim();

  TORCH_CHECK(dim_tensor1 != 0 && dim_tensor2 != 0,
              "both arguments to matmul need to be at least 1D, but they are ",
              dim_tensor1, "D and ", dim_tensor2, "D");

  const bool has_out = out.defined();

  if (has_out) {
    TORCH_CHECK(!(tensor1.requires_grad() || tensor2.requires_grad() || out.requires_grad()) || !at::GradMode::is_enabled(),
      "matmul(): functions with out=... arguments don't support automatic differentiation, "
      "but one of the arguments requires grad."
    );
  }

  if (dim_tensor1 == 1 && dim_tensor2 == 1) {
    return has_out ? at::dot_out(out, tensor1, tensor2) : tensor1.dot(tensor2);
  } else if (dim_tensor1 == 2 && dim_tensor2 == 1) {
    return has_out ? at::mv_out(out, tensor1, tensor2) : tensor1.mv(tensor2);
  } else if (dim_tensor1 == 1 && dim_tensor2 == 2) {
    return has_out ? at::mm_out(out, tensor1.unsqueeze(0), tensor2).squeeze_(0)
                   : tensor1.unsqueeze(0).mm(tensor2).squeeze_(0);
  } else if (dim_tensor1 == 2 && dim_tensor2 == 2) {
    return has_out ? at::mm_out(out, tensor1, tensor2) : tensor1.mm(tensor2);
  } else if (should_fold(tensor1, tensor2, has_out)) {
    const auto transpose = dim_tensor2 > dim_tensor1;
    const auto t1 = transpose ? MaybeOwned<Tensor>::owned(tensor2.mT())
                              : MaybeOwned<Tensor>::borrowed(tensor1);
    const auto t2 = !transpose ? MaybeOwned<Tensor>::borrowed(tensor2)
                               : dim_tensor1 == 2
                                   ? MaybeOwned<Tensor>::owned(tensor1.t())
                                   : MaybeOwned<Tensor>::borrowed(tensor1);

    const auto sizes_1 = t1->sizes();
    auto output_shape = DimVector(sizes_1.begin(), sizes_1.end() - 1);
    const auto folded_dim1 = c10::multiply_integers(output_shape);

    const auto t2_is_matrix = t2->dim() == 2;
    if (t2_is_matrix) {
      output_shape.push_back(t2->sizes()[1]);
    }
    const auto t1_folded = t1->reshape({folded_dim1, sizes_1.back()});
    if (!has_out) {
      if (t2_is_matrix) {
        const auto output = at::_unsafe_view(t1_folded.mm(*t2), output_shape);
        return transpose ? output.mT().contiguous() : output;
      } else {
        return at::_unsafe_view(t1_folded.mv(*t2), output_shape);
      }
    } else {
      TORCH_INTERNAL_ASSERT(!(transpose && t2_is_matrix));
      at::native::resize_output(out, output_shape);
      auto reshaped_out = t2_is_matrix ? out.reshape({folded_dim1, t2->sizes().back()})
                                       : out.reshape({folded_dim1});
      if (t2_is_matrix) {
        at::mm_out(reshaped_out, t1_folded, *t2);
      } else {
        at::mv_out(reshaped_out, t1_folded, *t2);
      }
      if (!reshaped_out.is_alias_of(out)) {
        out.copy_(reshaped_out);
      }
      return out;
    }
  } else {
    const int64_t n = dim_tensor1 > 1 ? tensor1.sizes().cend()[-2] : 1LL;
    const int64_t m1 = tensor1.sizes().back();
    auto batch_tensor1 = tensor1.sizes().slice(0, std::max<int64_t>(dim_tensor1 - 2, 0LL));
    const int64_t m2 = dim_tensor2 > 1 ? tensor2.sizes().cend()[-2] : tensor2.sizes().front();
    const int64_t p = dim_tensor2 > 1 ? tensor2.sizes().back() : 1LL;
    const IntArrayRef batch_tensor2(tensor2.sizes().data(),
                                    std::max<int64_t>(dim_tensor2 - 2, 0LL));

    if (dim_tensor1 == 3 && dim_tensor2 == 3 && batch_tensor1[0] != batch_tensor2[0]) {
      if (batch_tensor1[0] == 1 && (tensor1.requires_grad() || isTensorSubclassLike(tensor1))) {
        return _matmul_impl(out, tensor1.squeeze(0), tensor2);
      }
      if (batch_tensor2[0] == 1 && (tensor2.requires_grad() || isTensorSubclassLike(tensor2))) {
        return _matmul_impl(out, tensor1, tensor2.squeeze(0));
      }
    }

    auto output_shape = infer_size_dimvector(batch_tensor1, batch_tensor2);
    const int64_t expand_batch_product = c10::multiply_integers(output_shape);

    const auto tensor1_expand_size = [&output_shape, n, m1]{ DimVector ret(output_shape);
                                                             ret.append({n, m1});
                                                             return ret; }();
    const auto tensor1_expanded = tensor1.expand(tensor1_expand_size)
                                         .reshape({expand_batch_product, n, m1});
    auto vector_rhs = dim_tensor2 == 1;
    const auto tensor2_expand_size = [&output_shape, m2, p, vector_rhs]{
      DimVector ret(output_shape);
      if (vector_rhs) {
        ret.push_back(m2);
      } else {
        ret.append({m2, p});
      }
      return ret;
    }();
    auto tensor2_expanded = tensor2.expand(tensor2_expand_size);
    if (vector_rhs) {
      tensor2_expanded = tensor2_expanded.reshape({expand_batch_product, m2}).unsqueeze(2);
    } else {
      tensor2_expanded = tensor2_expanded.reshape({expand_batch_product, m2, p});
    }

    if (dim_tensor1 > 1) {
      output_shape.push_back(n);
    }
    if (dim_tensor2 > 1) {
      output_shape.push_back(p);
    }

    if (!has_out) {
      if (vector_rhs) {
        return at::_unsafe_view(tensor1_expanded.bmm(tensor2_expanded).squeeze(-1), output_shape);
      } else {
        return at::_unsafe_view(tensor1_expanded.bmm(tensor2_expanded), output_shape);
      }
    } else {
      at::native::resize_output(out, output_shape);
      auto reshaped_out = out.reshape({expand_batch_product, n, p});
      at::bmm_out(reshaped_out, tensor1_expanded, tensor2_expanded);
      if (vector_rhs) {
        reshaped_out = reshaped_out.squeeze(-1);
      }
      if (!reshaped_out.is_alias_of(out)) {
        out.copy_(reshaped_out.view_as(out));
      }
      return out;
    }
  }
}

Tensor matmul(const Tensor & tensor1, const Tensor & tensor2) {
  auto maybe_outnames = namedinference::compute_matmul_outnames(tensor1, tensor2);
  at::Tensor result, unused;
  result = at::native::_matmul_impl(unused, tensor1, tensor2);
  namedinference::propagate_names_if_nonempty(result, maybe_outnames);
  return result;
}

Tensor& matmul_out(const Tensor & tensor1, const Tensor & tensor2, Tensor &result) {
  auto maybe_outnames = namedinference::compute_matmul_outnames(tensor1, tensor2);
  at::native::_matmul_impl(result, tensor1, tensor2);
  namedinference::propagate_names_if_nonempty(result, maybe_outnames);
  return result;
}

std::tuple<Tensor, Tensor> matmul_backward(
    const Tensor& grad,
    const Tensor& self,
    const Tensor& other,
    std::array<bool, 2> mask) {
  Tensor self_grad, other_grad;
  if (mask[0]) {
    self_grad = at::matmul(grad, other.mT());
  }
  if (mask[1]) {
    other_grad = at::matmul(self.mT(), grad);
  }
  return std::make_tuple(std::move(self_grad), std::move(other_grad));
}

} // namespace at::native
