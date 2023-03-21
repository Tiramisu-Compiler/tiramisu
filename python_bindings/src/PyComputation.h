#ifndef TIRAMISU_PYTHON_BINDINGS_PYCOMP_H
#define TIRAMISU_PYTHON_BINDINGS_PYCOMP_H
#include "PyTiramisu.h"

namespace tiramisu {
  namespace PythonBindings {

    void define_computation(py::module &m);

  }  // namespace PythonBindings
}  // namespace tiramisu

#endif // TIRAMISU_PYTHON_BINDINGS_PYCOMP_H
