#include <ATen/Context.h>
#include <c10/core/ScalarType.h>
#include <c10/util/irange.h>
#include <torch/csrc/autograd/generated/VariableType.h>

namespace torch {
namespace autograd::VariableType {

static std::vector<at::DeprecatedTypeProperties*> allTypesForBackends(
    at::ArrayRef<at::Backend> backends) {
  std::vector<at::DeprecatedTypeProperties*> res;
  res.reserve(backends.size());
  for (auto backend : backends) {
    for (const auto scalar_type :
         c10::irange(static_cast<int64_t>(at::ScalarType::NumOptions))) {
      auto& type = at::getDeprecatedTypeProperties(
          static_cast<at::Backend>(backend),
          static_cast<at::ScalarType>(scalar_type));
      res.emplace_back(&type);
    }
  }
  return res;
}

std::vector<at::DeprecatedTypeProperties*> allCPUTypes() {
  return allTypesForBackends({at::Backend::CPU, at::Backend::SparseCPU});
}

std::vector<at::DeprecatedTypeProperties*> allCUDATypes() {
  at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);
  return allTypesForBackends({at::Backend::CUDA, at::Backend::SparseCUDA});
}

std::vector<at::DeprecatedTypeProperties*> allXPUTypes() {
  return allTypesForBackends({at::Backend::XPU, at::Backend::SparseXPU});
}

std::vector<at::DeprecatedTypeProperties*> allPrivateUser1Types() {
  at::globalContext().lazyInitDevice(c10::DeviceType::PrivateUse1);
  return allTypesForBackends(
      {at::Backend::PrivateUse1, at::Backend::SparsePrivateUse1});
}

} // namespace autograd::VariableType
} // namespace torch
