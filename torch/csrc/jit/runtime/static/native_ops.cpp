#include <torch/csrc/jit/runtime/static/ops.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch::jit {

C10_DEFINE_REGISTRY(SRNativeOperatorRegistry, SROperatorFunctor)

bool nativeOpIsRegistered(const c10::Symbol& op_name) {
  const std::string name(op_name.toQualString());
  return SRNativeOperatorRegistry()->Has(name);
}

SROperator getNativeOperation(Node* n) {
  auto op_name = n->kind().toQualString();
  if (SRNativeOperatorRegistry()->Has(op_name)) {
    return SRNativeOperatorRegistry()->Create(op_name)->Generate(n);
  }
  return nullptr;
}

} // namespace torch::jit
