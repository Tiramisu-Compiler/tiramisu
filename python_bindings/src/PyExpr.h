#ifndef TIRAMISU_PYTHON_BINDINGS_PYEXPR_H
#define TIRAMISU_PYTHON_BINDINGS_PYEXPR_H
#include "PyTiramisu.h"

namespace tiramisu {
  namespace PythonBindings {

    void define_expr(py::module &m);

  }  // namespace PythonBindings
}  // namespace tiramisu

#endif  // TIRAMISU_PYTHON_BINDINGS_PYEXPR_H
