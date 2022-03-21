#ifndef TIRAMISU_PYTHON_BINDINGS_PYINIT_H
#define TIRAMISU_PYTHON_BINDINGS_PYINIT_H
#include "PyTiramisu.h"

namespace tiramisu {
  namespace PythonBindings {

    void define_init(py::module &m);

  }  // namespace PythonBindings
}  // namespace tiramisu

#endif  // TIRAMISU_PYTHON_BINDINGS_PYINIT_H