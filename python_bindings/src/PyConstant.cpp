#include "PyConstant.h"

namespace tiramisu {
  namespace PythonBindings {

    void define_const(py::module &m){
      auto const_class = py::class_<constant>(m, "constant")
        .def(py::init<std::string, const tiramisu::expr &>());
    }

  }  // namespace PythonBindings
}  // namespace tiramisu
