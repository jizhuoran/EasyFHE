#include <torch/csrc/python_headers.h>

bool THPEngine_initModule(PyObject* module) {
  PyObject* none = Py_None;
  Py_INCREF(none);
  return PyModule_AddObject(module, "_ImperativeEngine", none) == 0;
}
