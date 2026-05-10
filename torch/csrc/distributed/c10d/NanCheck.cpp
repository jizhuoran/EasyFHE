#include <torch/csrc/distributed/c10d/NanCheck.hpp>

namespace c10d {

void checkForNan(const at::Tensor&) {}

} // namespace c10d
