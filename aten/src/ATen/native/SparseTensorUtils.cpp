#include <ATen/native/sparse/SparseStubs.h>

namespace at::native {

DEFINE_DISPATCH(flatten_indices_stub);
DEFINE_DISPATCH(mul_sparse_sparse_out_stub);
DEFINE_DISPATCH(sparse_mask_intersection_out_stub);
DEFINE_DISPATCH(sparse_mask_projection_out_stub);

} // namespace at::native
