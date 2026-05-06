#include <torch/csrc/jit/runtime/static/ops.h>
#include <torch/csrc/jit/ir/ir.h>
#include <c10/util/irange.h>

C10_DEFINE_bool(
    static_runtime_enable_fast_math,
    true,
    "If on, static runtime may use optimizations that cause accuracy loss "
    "vs the jit interpreter")

namespace torch::jit {

C10_DEFINE_REGISTRY(SROperatorRegistry, SROperatorFunctor)

bool opIsRegistered(const c10::Symbol& op_name) {
  const std::string name(op_name.toQualString());
  return SROperatorRegistry()->Has(name);
}

SROperator getOutOfPlaceOperation(Node* n) {
  auto op_name = n->kind().toQualString();
  if (SROperatorRegistry()->Has(op_name)) {
    return SROperatorRegistry()->Create(op_name)->Generate(n);
  }
  return nullptr;
}

bool hasVarArgs(Node* n) {
  if (n->kind() == prim::VarConcat || n->kind() == prim::VarStack) {
    return true;
  }
  return false;
}

bool canReuseInputsOutputs(
    Node* n,
    const c10::FastMap<Node*, bool>& node_has_out_variant) {
  auto it = node_has_out_variant.find(n);
  if (it != node_has_out_variant.end()) {
    return it->second;
  }
  return getOutOfPlaceOperation(n) != nullptr;
}

static bool inputsCanRunOutOfPlace(
    Node* n,
    const c10::FastMap<Node*, bool>& node_has_out_variant) {
  for (auto* input : n->inputs()) {
    if (!canReuseInputsOutputs(input->node(), node_has_out_variant)) {
      return false;
    }
  }
  return true;
}

bool isOptimizableContainerType(
    Node* n,
    const c10::FastMap<Node*, bool>& node_has_out_variant) {
  const auto& type = n->output()->type();
  bool is_supported_type = false;
  if (type->kind() == TypeKind::ListType) {
    const auto& list_type = type->expectRef<ListType>();
    is_supported_type =
        list_type.getElementType()->kind() == TypeKind::TensorType;
  } else if (type->kind() == TypeKind::TupleType) {
    const auto& tuple_type = type->expectRef<TupleType>();
    auto types = tuple_type.containedTypes();
    const auto& iter =
        std::find_if(types.begin(), types.end(), [](const TypePtr& elem) {
          return elem->kind() == TypeKind::TensorType;
        });
    is_supported_type = iter != types.end();
  }
  return is_supported_type && inputsCanRunOutOfPlace(n, node_has_out_variant);
}

} // namespace torch::jit
