#ifndef TIRAMISU_PYTHON_BINDINGS_PYTYPE_H
#define TIRAMISU_PYTHON_BINDINGS_PYTYPE_H
#include "PyTiramisu.h"
#include <tiramisu/type.h>
namespace tiramisu {
  namespace PythonBindings {

    void define_type(py::module &m);

  }  // namespace PythonBindings
}  // namespace tiramisu

#endif  // TIRAMISU_PYTHON_BINDINGS_PYTYPE_H
