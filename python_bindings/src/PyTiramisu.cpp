#include "PyTiramisu.h"
#include "PyExpr.h"
static_assert(PYBIND11_VERSION_MAJOR == 2 && PYBIND11_VERSION_MINOR >= 6,
              "Halide requires PyBind 2.6+");

static_assert(PY_VERSION_HEX >= 0x03000000,
              "We appear to be compiling against Python 2.x rather than 3.x, which is not supported.");


#ifndef TIRAMISU_PYBIND_MODULE_NAME
#define TIRAMISU_PYBIND_MODULE_NAME tiramisu
#endif

PYBIND11_MODULE(TIRAMISU_PYBIND_MODULE_NAME, m) {
  using namespace tiramisu::PythonBindings;
  define_expr(m);
}
