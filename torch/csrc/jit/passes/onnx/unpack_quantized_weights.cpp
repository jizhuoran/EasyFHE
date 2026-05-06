#include <torch/csrc/jit/passes/onnx/unpack_quantized_weights.h>

namespace torch::jit {

void UnpackQuantizedWeights(
    std::shared_ptr<Graph>& graph,
    std::map<std::string, IValue>& paramsDict) {
}

void insertPermutes(
    std::shared_ptr<Graph>& graph,
    std::map<std::string, IValue>& paramsDict) {
}

} // namespace torch::jit
