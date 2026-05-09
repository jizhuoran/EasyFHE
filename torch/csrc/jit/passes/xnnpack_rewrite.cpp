#include <torch/csrc/jit/passes/xnnpack_rewrite.h>

namespace torch::jit {

void transformConv1dToConv2d(std::shared_ptr<Graph>&) {}

void transformConv1dToConv2d(script::Module&) {}

void insertPrePackedOps(std::shared_ptr<Graph>&) {}

void insertPrePackedOps(script::Module&) {}

void fusePrePackedLinearConvWithClamp(script::Module&) {}

void FoldPrePackingOps(script::Module&) {}

script::Module optimizeForMobile(
    const script::Module& module,
    const std::set<MobileOptimizerType>&,
    const std::vector<std::string>&) {
  return module;
}

} // namespace torch::jit
