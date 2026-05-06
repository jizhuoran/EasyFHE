#include <ATen/native/RNN.h>

namespace at::native {

DEFINE_DISPATCH(lstm_cudnn_stub);
DEFINE_DISPATCH(lstm_miopen_stub);
DEFINE_DISPATCH(lstm_mkldnn_stub);
DEFINE_DISPATCH(gru_cudnn_stub);
DEFINE_DISPATCH(gru_miopen_stub);
DEFINE_DISPATCH(rnn_tanh_cudnn_stub);
DEFINE_DISPATCH(rnn_tanh_miopen_stub);
DEFINE_DISPATCH(rnn_relu_cudnn_stub);
DEFINE_DISPATCH(rnn_relu_miopen_stub);
DEFINE_DISPATCH(lstm_packed_cudnn_stub);
DEFINE_DISPATCH(lstm_packed_miopen_stub);
DEFINE_DISPATCH(gru_packed_cudnn_stub);
DEFINE_DISPATCH(gru_packed_miopen_stub);
DEFINE_DISPATCH(rnn_tanh_packed_cudnn_stub);
DEFINE_DISPATCH(rnn_tanh_packed_miopen_stub);
DEFINE_DISPATCH(rnn_relu_packed_cudnn_stub);
DEFINE_DISPATCH(rnn_relu_packed_miopen_stub);

} // namespace at::native
