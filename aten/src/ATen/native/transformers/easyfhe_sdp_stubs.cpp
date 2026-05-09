#include <ATen/native/transformers/cuda/sdp_utils.h>

namespace sdp {

bool is_flash_attention_available() {
  return false;
}

bool can_use_flash_attention(sdp_params const& params, bool debug) {
  return false;
}

bool can_use_mem_efficient_attention(sdp_params const& params, bool debug) {
  return false;
}

bool can_use_cudnn_attention(sdp_params const& params, bool debug) {
  return false;
}

} // namespace sdp
