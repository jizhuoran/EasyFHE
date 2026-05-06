#include <ATen/native/BatchLinearAlgebra.h>

namespace at::native {

DEFINE_DISPATCH(cholesky_stub);
DEFINE_DISPATCH(cholesky_inverse_stub);
DEFINE_DISPATCH(geqrf_stub);
DEFINE_DISPATCH(ldl_factor_stub);
DEFINE_DISPATCH(ldl_solve_stub);
DEFINE_DISPATCH(linalg_eig_stub);
DEFINE_DISPATCH(linalg_eigh_stub);
DEFINE_DISPATCH(lstsq_stub);
DEFINE_DISPATCH(lu_factor_stub);
DEFINE_DISPATCH(lu_solve_stub);
DEFINE_DISPATCH(orgqr_stub);
DEFINE_DISPATCH(ormqr_stub);
DEFINE_DISPATCH(svd_stub);
DEFINE_DISPATCH(triangular_solve_stub);
DEFINE_DISPATCH(unpack_pivots_stub);
DEFINE_DISPATCH(linalg_eig_make_complex_eigenvectors_stub);

} // namespace at::native
