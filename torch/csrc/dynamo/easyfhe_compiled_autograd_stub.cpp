#include <torch/csrc/dynamo/compiled_autograd.h>

namespace torch::dynamo::autograd {
namespace {
std::unique_ptr<PyCompilerInterface> compiler_interface;
} // namespace

const std::unique_ptr<PyCompilerInterface>& getPyCompilerInterface() {
  return compiler_interface;
}

PyCompilerGuard::PyCompilerGuard(std::unique_ptr<PyCompilerInterface>&& impl) {
  compiler_interface = std::move(impl);
}

PyCompilerGuard::~PyCompilerGuard() {
  compiler_interface.reset();
}

std::vector<std::optional<InputMetadata>> get_input_metadata(
    const edge_list& edges) {
  return std::vector<std::optional<InputMetadata>>(edges.size());
}

} // namespace torch::dynamo::autograd
