#ifndef TIRAMISU_PYTHON_BINDINGS_PYBUFFER_H
#define TIRAMISU_PYTHON_BINDINGS_PYBUFFER_H
#include "PyTiramisu.h"

namespace tiramisu {
  namespace PythonBindings {

    void define_buffer(py::module &m);

  }  // namespace PythonBindings
}  // namespace tiramisu

#endif  // TIRAMISU_PYTHON_BINDINGS_PYBUFFER_H