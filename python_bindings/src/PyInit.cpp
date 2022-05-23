#include "PyInit.h"
#include "../../include/tiramisu/core.h"

namespace tiramisu {
  namespace PythonBindings {

    void init_py(std::string name) {
        tiramisu::init(name);
    }

    void define_init(py::module &m){
      m.def("init", &init_py, "Set up Tiramisu default function", py::return_value_policy::reference);
      m.def("get_implicit_function", &global::get_implicit_function, py::return_value_policy::reference);
    }

  }  // namespace PythonBindings
}  // namespace tiramisu
