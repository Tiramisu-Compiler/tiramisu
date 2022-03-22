#include "PyConstant.h"

namespace tiramisu {
  namespace PythonBindings {

    void define_const(py::module &m){
      auto const_class = py::class_<constant>(m, "constant")
        .def(py::init<std::string, const tiramisu::expr &>());

      py::implicitly_convertible<tiramisu::constant, tiramisu::expr>();
    }

  }  // namespace PythonBindings
}  // namespace tiramisu
