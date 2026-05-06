#include <ATen/native/SegmentReduce.h>

namespace at::native {

DEFINE_DISPATCH(_segment_reduce_lengths_stub);
DEFINE_DISPATCH(_segment_reduce_offsets_stub);
DEFINE_DISPATCH(_segment_reduce_lengths_backward_stub);
DEFINE_DISPATCH(_segment_reduce_offsets_backward_stub);

} // namespace at::native
