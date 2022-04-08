#ifndef TIRAMISU_PYTHON_BINDINGS_PYFUNCTION_H
#define TIRAMISU_PYTHON_BINDINGS_PYFUNCTION_H
#include "PyTiramisu.h"

namespace tiramisu {
  namespace PythonBindings {

    void define_function(py::module &m);

  }  // namespace PythonBindings
}  // namespace tiramisu

#endif // TIRAMISU_PYTHON_BINDINGS_PYFUNCTION_H
