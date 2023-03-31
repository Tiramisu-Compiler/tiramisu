#include "PyConstant.h"

namespace tiramisu
{
  namespace PythonBindings
  {

    void define_const(py::module &m)
    {
      auto const_class = py::class_<constant, tiramisu::computation>(m, "constant")
                             .def(py::init<std::string, const tiramisu::expr &>(),
                                  py::return_value_policy::reference, py::keep_alive<0, 2>())
                             .def(py::init([]( // Create new constructor that doesn't use the last 2 args to avoid having to add the default values in python
                                               std::string param_name,
                                               const tiramisu::expr &param_expr,
                                               tiramisu::primitive_t t,
                                               bool function_wide,
                                               tiramisu::computation *with_computation,
                                               int at_loop_level
                                               // tiramisu::function,// *fct,
                                           )
                                           { return new constant(param_name, param_expr, t, function_wide, with_computation, at_loop_level); }),
                                  py::keep_alive<0, 2>(), py::keep_alive<0, 5>());
    }

  } // namespace PythonBindings
} // namespace tiramisu
