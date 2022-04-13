#include "PyFunction.h"

namespace tiramisu {
  namespace PythonBindings {
    void define_function(py::module &m){
      auto function_class = py::class_<function>(m, "function")
	.def(py::init<std::string>())
	.def("dump", &function::dump)
	.def("codegen", py::overload_cast<const std::vector<tiramisu::buffer *> &, const std::string, const bool, bool>(&tiramisu::function::codegen));
      //Printer, add buffers, get buffers?

    }


  }  // namespace PythonBindings
}  // namespace tiramisu




