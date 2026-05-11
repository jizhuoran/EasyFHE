#include <torch/csrc/autograd/autograd.h>

namespace torch::autograd {

void backward(
    const variable_list& tensors,
    const variable_list& grad_tensors,
    std::optional<bool> retain_graph,
    bool create_graph,
    const variable_list& inputs) {
  TORCH_CHECK(false, "torch.autograd backward engine is disabled in EasyFHE");
}

variable_list grad(
    const variable_list& outputs,
    const variable_list& inputs,
    const variable_list& grad_outputs,
    std::optional<bool> retain_graph,
    bool create_graph,
    bool allow_unused) {
  TORCH_CHECK(false, "torch.autograd grad engine is disabled in EasyFHE");
}

namespace forward_ad {

uint64_t enter_dual_level() {
  TORCH_CHECK(false, "forward-mode autograd is disabled in EasyFHE");
}

void exit_dual_level(uint64_t level) {
  TORCH_CHECK(false, "forward-mode autograd is disabled in EasyFHE");
}

} // namespace forward_ad

} // namespace torch::autograd
