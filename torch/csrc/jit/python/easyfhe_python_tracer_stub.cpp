#include <torch/csrc/python_headers.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/jit/python/python_tracer.h>

#include <c10/util/Exception.h>

namespace torch::jit::tracer {

void initPythonTracerBindings(PyObject*) {}

Node* preRecordPythonTrace(
    THPObjectPtr,
    const std::string&,
    at::ArrayRef<autograd::Variable>,
    std::vector<THPObjectPtr>) {
  TORCH_CHECK(false, "JIT Python tracing is disabled in EasyFHE fast build");
}

} // namespace torch::jit::tracer

namespace torch::jit {

IValue toIValue(py::handle obj, const TypePtr&, std::optional<int32_t>) {
  if (THPVariable_Check(obj.ptr())) {
    return IValue(THPVariable_Unpack(obj.ptr()));
  }
  if (obj.is_none()) {
    return IValue();
  }
  if (py::isinstance<py::bool_>(obj)) {
    return IValue(py::cast<bool>(obj));
  }
  if (py::isinstance<py::int_>(obj)) {
    return IValue(py::cast<int64_t>(obj));
  }
  if (py::isinstance<py::float_>(obj)) {
    return IValue(py::cast<double>(obj));
  }
  if (py::isinstance<py::str>(obj)) {
    return IValue(py::cast<std::string>(obj));
  }
  TORCH_CHECK(false, "JIT Python IValue conversion is disabled in EasyFHE fast build");
}

py::object toPyObject(IValue ivalue) {
  if (ivalue.isNone()) {
    return py::none();
  }
  if (ivalue.isTensor()) {
    return py::reinterpret_steal<py::object>(THPVariable_Wrap(ivalue.toTensor()));
  }
  if (ivalue.isBool()) {
    return py::bool_(ivalue.toBool());
  }
  if (ivalue.isInt()) {
    return py::int_(ivalue.toInt());
  }
  if (ivalue.isDouble()) {
    return py::float_(ivalue.toDouble());
  }
  if (ivalue.isString()) {
    return py::str(ivalue.toStringRef());
  }
  TORCH_CHECK(false, "JIT Python IValue conversion is disabled in EasyFHE fast build");
}

} // namespace torch::jit

namespace torch::jit::detail {

std::optional<InferredType> _tryToInferTypeImpl(py::handle) {
  return std::nullopt;
}

} // namespace torch::jit::detail
