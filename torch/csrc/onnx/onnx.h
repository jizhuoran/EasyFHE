#pragma once

namespace torch::onnx {

enum class OperatorExportTypes {
  ONNX,
  ONNX_ATEN,
  ONNX_ATEN_FALLBACK,
  ONNX_FALLTHROUGH,
};

enum class TrainingMode {
  EVAL,
  PRESERVE,
  TRAINING,
};

constexpr auto kOnnxNodeNameAttribute = "onnx_name";

} // namespace torch::onnx
