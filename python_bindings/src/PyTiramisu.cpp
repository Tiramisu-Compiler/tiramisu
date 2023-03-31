#include "PyTiramisu.h"
#include "PyType.h"
#include "PyExpr.h"
#include "PyBuffer.h"
#include "PyInit.h"
#include "PyConstant.h"
#include "PyVar.h"
#include "PyComputation.h"
#include "PyCodegen.h"
#include "PyInput.h"
#include "PyFunction.h"
static_assert(PYBIND11_VERSION_MAJOR == 2 && PYBIND11_VERSION_MINOR >= 6,
              "Halide requires PyBind 2.6+");

static_assert(PY_VERSION_HEX >= 0x03000000,
              "We appear to be compiling against Python 2.x rather than 3.x, which is not supported.");

#ifndef TIRAMISU_PYBIND_MODULE_NAME
#define TIRAMISU_PYBIND_MODULE_NAME tiramisu
#endif

PYBIND11_MODULE(TIRAMISU_PYBIND_MODULE_NAME, m)
{
  using namespace tiramisu::PythonBindings;
  define_type(m);
  define_expr(m);
  define_buffer(m);
  define_init(m);
  define_computation(m);
  define_const(m);
  define_var(m);
  define_codegen(m);
  define_input(m);
  define_function(m);
}
