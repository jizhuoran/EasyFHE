#include <torch/csrc/mps/Module.h>

namespace torch::mps {

PyMethodDef* python_functions() {
  static PyMethodDef methods[] = {
      {"_mps_is_available",
       [](PyObject*, PyObject*) -> PyObject* {
         Py_RETURN_FALSE;
       },
       METH_NOARGS,
       nullptr},
      {nullptr, nullptr, 0, nullptr},
  };
  return methods;
}

void initModule(PyObject* module) {
  (void)module;
}

} // namespace torch::mps
