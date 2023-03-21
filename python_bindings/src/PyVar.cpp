#include "PyVar.h"

namespace tiramisu {
  namespace PythonBindings {

    void define_var(py::module &m){
      auto var_class = py::class_<var, expr>(m, "var")
        .def(py::init<tiramisu::primitive_t, std::string>(), py::return_value_policy::reference)
        .def(py::init<std::string>(), py::return_value_policy::reference)
        .def(py::init<std::string, tiramisu::expr, tiramisu::expr>(), py::return_value_policy::reference)
        .def(py::init<>(), py::return_value_policy::reference)
        .def("get_name", &var::get_name, py::return_value_policy::reference)
        .def("get_upper", &var::get_upper, py::return_value_policy::reference)
        .def("get_lower", &var::get_lower, py::return_value_policy::reference)
        .def("dump", [](const tiramisu::var &e) -> auto { return e.dump(true); });

      //TODO missing inits from tiramisu_expr.ccp
    }

  }  // namespace PythonBindings
}  // namespace tiramisu
