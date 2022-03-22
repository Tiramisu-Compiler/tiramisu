#include "PyExpr.h"

namespace tiramisu {
  namespace PythonBindings {

    void define_expr(py::module &m){
      auto expr_class = py::class_<expr>(m, "expr").def(py::init<>())
	      .def(py::init<primitive_t>())
        // for implicitly_convertible
        .def(py::init([](const tiramisu::constant &c) -> tiramisu::expr { return c.expr(); }));
      //operator
      //casts
      //vars
      //buffer
      //cuda syncrnize

      

      py::implicitly_convertible<tiramisu::constant, tiramisu::expr>();
      py::implicitly_convertible<tiramisu::primitive_t, tiramisu::expr>();
    }

  }  // namespace PythonBindings
}  // namespace tiramisu
