#include "PyInit.h"
#include "../../include/tiramisu/core.h"

namespace tiramisu {
  namespace PythonBindings {

    void define_codegen(py::module &m){
      m.def("codegen", 
            py::overload_cast<const std::vector<tiramisu::buffer *> &, const std::string, const bool>(&tiramisu::codegen));
    }

  }  // namespace PythonBindings
}  // namespace tiramisu
