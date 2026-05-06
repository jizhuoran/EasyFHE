#include <ATen/core/Tensor.h>

namespace at::native {

Tensor indices_default(const Tensor& self) {
  TORCH_CHECK(false, "indices expected sparse coordinate tensor layout but got ", self.layout());
}

Tensor values_default(const Tensor& self) {
  TORCH_CHECK(false, "values expected sparse tensor layout but got ", self.layout());
}

Tensor crow_indices_default(const Tensor& self) {
  TORCH_CHECK(false, "crow_indices expected sparse row compressed tensor layout but got ", self.layout());
}

Tensor col_indices_default(const Tensor& self) {
  TORCH_CHECK(false, "col_indices expected sparse row compressed tensor layout but got ", self.layout());
}

Tensor ccol_indices_default(const Tensor& self) {
  TORCH_CHECK(false, "ccol_indices expected sparse column compressed tensor layout but got ", self.layout());
}

Tensor row_indices_default(const Tensor& self) {
  TORCH_CHECK(false, "row_indices expected sparse column compressed tensor layout but got ", self.layout());
}

} // namespace at::native
