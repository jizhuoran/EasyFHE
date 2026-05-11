#include <torch/csrc/utils/python_symnode.h>

namespace torch {

py::handle get_symint_class() {
  // NB: leak
#if IS_PYBIND_2_13_PLUS
  PYBIND11_CONSTINIT static py::gil_safe_call_once_and_store<py::object>
      storage;
  return storage
      .call_once_and_store_result([]() -> py::object {
        return py::module::import("easyfhe").attr("SymInt");
      })
      .get_stored();
#else
  static py::handle symint_class =
      py::object(py::module::import("easyfhe").attr("SymInt")).release();
  return symint_class;
#endif
}

py::handle get_symfloat_class() {
  // NB: leak
#if IS_PYBIND_2_13_PLUS
  PYBIND11_CONSTINIT static py::gil_safe_call_once_and_store<py::object>
      storage;
  return storage
      .call_once_and_store_result([]() -> py::object {
        return py::module::import("easyfhe").attr("SymFloat");
      })
      .get_stored();
#else
  static py::handle symfloat_class =
      py::object(py::module::import("easyfhe").attr("SymFloat")).release();
  return symfloat_class;
#endif
}

py::handle get_symbool_class() {
  // NB: leak
#if IS_PYBIND_2_13_PLUS
  PYBIND11_CONSTINIT static py::gil_safe_call_once_and_store<py::object>
      storage;
  return storage
      .call_once_and_store_result([]() -> py::object {
        return py::module::import("easyfhe").attr("SymBool");
      })
      .get_stored();
#else
  static py::handle symbool_class =
      py::object(py::module::import("easyfhe").attr("SymBool")).release();
  return symbool_class;
#endif
}

py::handle get_dynint_class() {
  // NB: leak
#if IS_PYBIND_2_13_PLUS
  PYBIND11_CONSTINIT static py::gil_safe_call_once_and_store<py::object>
      storage;
  return storage
      .call_once_and_store_result([]() -> py::object {
        return py::module::import("easyfhe.fx.experimental.sym_node")
            .attr("DynamicInt");
      })
      .get_stored();
#else
  static py::handle symbool_class =
      py::object(py::module::import("easyfhe.fx.experimental.sym_node")
                     .attr("DynamicInt"))
          .release();
  return symbool_class;
#endif
}

} // namespace torch
