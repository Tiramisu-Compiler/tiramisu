#include "PyExpr.h"

namespace tiramisu {
  namespace PythonBindings {

    void define_expr(py::module &m){
      auto expr_class = py::class_<expr>(m, "expr").def(py::init<>())
	.def(py::init<primitive_t>());
      //operator
      //casts
      //vars
      //buffer
      //cuda syncrnize
    }

  }  // namespace PythonBindings
}  // namespace tiramisu
