#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/TensorOperators.h>
#include <c10/util/Exception.h>
#include <optional>

#include <ATen/CPUGeneratorImpl.h>
#include <ATen/core/DistributionsHelper.h>
#include <ATen/native/Distributions.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/DistributionTemplates.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/native/cpu/Loops.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/bernoulli_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/exponential_native.h>
#include <ATen/ops/normal_native.h>
#include <ATen/ops/random_native.h>
#include <ATen/ops/uniform_native.h>
#endif

#include <utility>

namespace at::native {

DEFINE_DISPATCH(bernoulli_tensor_stub);
DEFINE_DISPATCH(bernoulli_scalar_stub);
DEFINE_DISPATCH(exponential_stub);
DEFINE_DISPATCH(uniform_stub);
DEFINE_DISPATCH(normal_stub);
DEFINE_DISPATCH(random_stub);
DEFINE_DISPATCH(random_from_to_stub);
DEFINE_DISPATCH(random_full_64_bits_range_stub);
DEFINE_DISPATCH(cauchy_stub);
DEFINE_DISPATCH(geometric_stub);
DEFINE_DISPATCH(log_normal_stub);
DEFINE_DISPATCH(multinomial_with_replacement_stub);

// ==================================================== Bernoulli =====================================================

template<typename RNG>
struct BernoulliStub {
  void operator()(Tensor& self, const Tensor& p_, std::optional<Generator> gen) {
    bernoulli_tensor_stub(self.device().type(), self, p_, gen);
  }

  void operator()(Tensor& self, double p, std::optional<Generator> gen) {
    bernoulli_scalar_stub(self.device().type(), self, p, gen);
  }
};

Tensor bernoulli(const Tensor& self, std::optional<Generator> gen) {
  Tensor result = at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  result.bernoulli_(self, std::move(gen));
  return result;
}

Tensor bernoulli(const Tensor& self, double p, std::optional<Generator> gen) {
  Tensor result = at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  result.bernoulli_(p, std::move(gen));
  return result;
}

Tensor& bernoulli_out(const Tensor& self, std::optional<Generator> gen, Tensor& result) {
  return at::native::templates::bernoulli_out_impl<BernoulliStub, Generator>(result, self, std::move(gen));
}

Tensor& bernoulli_(Tensor& self, const Tensor& p_, std::optional<Generator> gen) {
  return at::native::templates::bernoulli_impl_<BernoulliStub, Generator>(self, p_, std::move(gen));
}

Tensor& bernoulli_(Tensor& self, double p, std::optional<Generator> gen) {
  return at::native::templates::bernoulli_impl_<BernoulliStub, Generator>(self, p, std::move(gen));
}

// ================================================== Exponential =====================================================

template<typename RNG>
struct ExponentialStub {
  void operator()(TensorIteratorBase& iter, double lambda, std::optional<Generator> gen) {
    exponential_stub(iter.device_type(), iter, lambda, gen);
  }
};

Tensor& exponential_(Tensor& self, double lambda, std::optional<Generator> gen) {
  return at::native::templates::exponential_impl_<ExponentialStub, Generator>(self, lambda, std::move(gen));
}

// ==================================================== Uniform =======================================================

template<typename RNG>
struct UniformStub {
  void operator()(TensorIteratorBase& iter, double from, double to, std::optional<Generator> gen) {
    uniform_stub(iter.device_type(), iter, from, to, gen);
  }
};

template<typename RNG>
struct UniformMeta {
  void operator()(TensorIteratorBase& iter, double from, double to, std::optional<Generator> gen) {
  }
};

Tensor& uniform_(Tensor& self, double from, double to, std::optional<Generator> gen) {
  return at::native::templates::uniform_impl_<UniformStub, Generator>(self, from, to, std::move(gen));
}

Tensor& uniform_meta_(Tensor& self, double from, double to, std::optional<Generator> gen) {
  return at::native::templates::uniform_impl_<UniformMeta, Generator>(self, from, to, std::move(gen));
}

// ==================================================== Normal ========================================================

template<typename RNG>
struct NormalStub {
  void operator()(Tensor& self, double mean, double std, std::optional<Generator> gen) {
    normal_stub(self.device().type(), self, mean, std, gen);
  }
};

template<typename RNG>
struct NormalMeta {
  void operator()(Tensor& self, double mean, double std, std::optional<Generator> gen) {
  }
};

Tensor& normal_(Tensor& self, double mean, double std, std::optional<Generator> gen) {
  return at::native::templates::normal_impl_<NormalStub, Generator>(self, mean, std, std::move(gen));
}

Tensor& normal_meta_(Tensor& self, double mean, double std, std::optional<Generator> gen) {
  return at::native::templates::normal_impl_<NormalMeta, Generator>(self, mean, std, std::move(gen));
}

Tensor& normal_out(const Tensor& mean, double std, std::optional<Generator> gen, Tensor& output) {
  return at::native::templates::normal_out_impl<NormalStub, Generator>(output, mean, std, std::move(gen));
}

Tensor& normal_out_meta(const Tensor& mean, double std, std::optional<Generator> gen, Tensor& output) {
  return at::native::templates::normal_out_impl<NormalMeta, Generator>(output, mean, std, std::move(gen));
}

Tensor& normal_out(double mean, const Tensor& std, std::optional<Generator> gen, Tensor& output) {
  return at::native::templates::normal_out_impl<NormalStub, Generator>(output, mean, std, std::move(gen));
}

Tensor& normal_out_meta(double mean, const Tensor& std, std::optional<Generator> gen, Tensor& output) {
  return at::native::templates::normal_out_impl<NormalMeta, Generator>(output, mean, std, std::move(gen));
}

Tensor& normal_out(const Tensor& mean, const Tensor& std, std::optional<Generator> gen, Tensor& output) {
  return at::native::templates::normal_out_impl<NormalStub, Generator>(output, mean, std, std::move(gen));
}

Tensor& normal_out_meta(const Tensor& mean, const Tensor& std, std::optional<Generator> gen, Tensor& output) {
  return at::native::templates::normal_out_impl<NormalMeta, Generator>(output, mean, std, std::move(gen));
}

Tensor normal(const Tensor& mean, double std, std::optional<Generator> gen) {
  return at::native::templates::normal_impl<NormalStub, Generator>(mean, std, std::move(gen));
}

Tensor normal_meta(const Tensor& mean, double std, std::optional<Generator> gen) {
  return at::native::templates::normal_impl<NormalMeta, Generator>(mean, std, std::move(gen));
}

Tensor normal(double mean, const Tensor& std, std::optional<Generator> gen) {
  return at::native::templates::normal_impl<NormalStub, Generator>(mean, std, std::move(gen));
}

Tensor normal_meta(double mean, const Tensor& std, std::optional<Generator> gen) {
  return at::native::templates::normal_impl<NormalMeta, Generator>(mean, std, std::move(gen));
}

Tensor normal(const Tensor& mean, const Tensor& std, std::optional<Generator> gen) {
  return at::native::templates::normal_impl<NormalStub, Generator>(mean, std, std::move(gen));
}

Tensor normal_meta(const Tensor& mean, const Tensor& std, std::optional<Generator> gen) {
  return at::native::templates::normal_impl<NormalMeta, Generator>(mean, std, std::move(gen));
}

Tensor normal_functional(const Tensor& self, double mean, double std, std::optional<at::Generator> generator) {
  return self.clone().normal_(mean, std, std::move(generator));
}

// ==================================================== Random ========================================================

template<typename RNG>
struct RandomStub {
  void operator()(TensorIteratorBase& iter, std::optional<Generator> gen) {
    random_stub(iter.device_type(), iter, gen);
  }
};

Tensor& random_(Tensor& self, std::optional<Generator> gen) {
  return at::native::templates::random_impl<RandomStub, Generator>(self, std::move(gen));
}

template<typename RNG>
struct RandomFromToStub {
  void operator()(TensorIteratorBase& iter, uint64_t range, int64_t from, std::optional<Generator> gen) {
    random_from_to_stub(iter.device_type(), iter, range, from, gen);
  }
  void operator()(TensorIteratorBase& iter, std::optional<Generator> gen) {
    random_full_64_bits_range_stub(iter.device_type(), iter, gen);
  }
};

Tensor& random_(Tensor& self, int64_t from, std::optional<int64_t> to, std::optional<Generator> gen) {
  return at::native::templates::random_from_to_impl<RandomFromToStub, Generator>(self, from, to, std::move(gen));
}

Tensor& random_(Tensor& self, int64_t to, std::optional<Generator> gen) {
  return random_(self, 0, to, std::move(gen));
}

Tensor& random_meta_(Tensor& self, std::optional<Generator> gen) {
  return self;
}

Tensor& random_meta_(Tensor& self, int64_t from, std::optional<int64_t> to, std::optional<Generator> gen) {
  return self;
}

Tensor& random_meta_(Tensor& self, int64_t to, std::optional<Generator> gen) {
  return self;
}

} // namespace at::native
