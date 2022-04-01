#ifndef TIRAMISU_PYTHON_BINDINGS_PYTIRAMISU_H
#define TIRAMISU_PYTHON_BINDINGS_PYTIRAMISU_H
#include <pybind11/numpy.h>
#include <pybind11/operators.h>


#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <string>
#include <vector>

#include <tiramisu/tiramisu.h>

namespace tiramisu {
namespace PythonBindings {
namespace py = pybind11;
    
} //namespace pythonbindings
} //namespace tiramisu

#endif // TIRAMISU_PYTHON_BINDINGS_PYTIRAMISU_H
