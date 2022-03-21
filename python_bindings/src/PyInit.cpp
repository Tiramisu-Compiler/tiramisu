#include "PyInit.h"
#include "../../include/tiramisu/core.h"

namespace tiramisu {
  namespace PythonBindings {

    void init_py(std::string name) {
        tiramisu::init(name);
    }

    void define_init(py::module &m){
      m.def("init", &init_py, "Set up Tiramisu default function");
    }

  }  // namespace PythonBindings
}  // namespace tiramisu
