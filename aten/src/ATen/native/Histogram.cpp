#include <ATen/native/Histogram.h>

namespace at::native {

DEFINE_DISPATCH(histogramdd_stub);
DEFINE_DISPATCH(histogramdd_linear_stub);
DEFINE_DISPATCH(histogram_select_outer_bin_edges_stub);

} // namespace at::native
