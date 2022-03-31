#include "PyExpr.h"

namespace tiramisu {
  namespace PythonBindings {

    void define_expr(py::module &m){
      auto expr_class = py::class_<expr>(m, "expr").def(py::init<>())
	      .def(py::init<primitive_t>())
        // for implicitly_convertible
	      .def(py::init<int>())
	      .def(py::init<double>())
        // constant convert
        .def(py::init([](tiramisu::constant &c) -> tiramisu::expr { return (tiramisu::expr) c; }))
        .def("dump", [](const tiramisu::expr &e) -> auto { return e.dump(true); })
        .def("__add__", [](tiramisu::expr &l, tiramisu::expr &r) -> auto { return l + r; });
      //operator
      //casts
      //vars
      //buffer
      //cuda syncrnize

      

      py::implicitly_convertible<tiramisu::constant, tiramisu::expr>();
      py::implicitly_convertible<int, tiramisu::expr>();
      py::implicitly_convertible<double, tiramisu::expr>();
    }

  }  // namespace PythonBindings
}  // namespace tiramisu
