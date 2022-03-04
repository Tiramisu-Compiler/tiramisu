#include "PyExpr.h"

namespace tiramisu {
  namespace PythonBindings {

    void define_expr(py::module &m){
      auto expr_class = py::class_<expr>(m, "expr").def(py::init<>());
    }

  }  // namespace PythonBindings
}  // namespace tiramisu
