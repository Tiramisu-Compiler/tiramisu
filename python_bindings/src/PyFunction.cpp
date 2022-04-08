#include "PyFunction.h"


namespace tiramisu {
  namespace PythonBindings {
    void define_function(py::module &m){
      auto function_class = py::class_<function>(m, "function");
    }


  }  // namespace PythonBindings
}  // namespace tiramisu




