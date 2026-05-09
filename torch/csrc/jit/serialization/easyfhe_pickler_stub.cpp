#include <torch/csrc/jit/serialization/pickler.h>

#include <c10/util/Exception.h>

namespace torch::jit {

Pickler::~Pickler() = default;

void Pickler::protocol() {
  constexpr char protocol[] = {static_cast<char>(0x80), static_cast<char>(2)};
  writer_(protocol, sizeof(protocol));
}

void Pickler::stop() {
  constexpr char stop[] = {'.'};
  writer_(stop, sizeof(stop));
}

void Pickler::pushIValue(const IValue&) {
  TORCH_CHECK(false, "JIT pickling is disabled in EasyFHE fast build");
}

} // namespace torch::jit
