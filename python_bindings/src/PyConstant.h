#ifndef TIRAMISU_PYTHON_BINDINGS_PYCONST_H
#define TIRAMISU_PYTHON_BINDINGS_PYCONST_H
#include "PyTiramisu.h"

namespace tiramisu {
  namespace PythonBindings {

    void define_const(py::module &m);

  }  // namespace PythonBindings
}  // namespace tiramisu

#endif  // TIRAMISU_PYTHON_BINDINGS_PYCONST_H