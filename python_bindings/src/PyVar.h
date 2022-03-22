#ifndef TIRAMISU_PYTHON_BINDINGS_PYVAR_H
#define TIRAMISU_PYTHON_BINDINGS_PYVAR_H
#include "PyTiramisu.h"

namespace tiramisu {
  namespace PythonBindings {

    void define_var(py::module &m);

  }  // namespace PythonBindings
}  // namespace tiramisu

#endif  // TIRAMISU_PYTHON_BINDINGS_PYVAR_H