#include "PyInit.h"
#include "../../include/tiramisu/core.h"

namespace tiramisu {
  namespace PythonBindings {

    void define_codegen(py::module &m){
      m.def("codegen", 
            py::overload_cast<const std::vector<tiramisu::buffer *> &, const std::string, const bool, bool>(&tiramisu::codegen),
            "This function generates the declared function and computations in an object file",
            py::arg("arguments"), py::arg("obj_filename"), py::arg("gen_cuda_stmt") = false, py::arg("gen_python") = false);
      
      m.def("codegen", 
            py::overload_cast<const std::vector<tiramisu::buffer *> &, const std::string, const tiramisu::hardware_architecture_t, bool>(&tiramisu::codegen),
            "This function generates the declared function and computations in an object file",
            py::arg("arguments"), py::arg("obj_filename"), py::arg("gen_architecture_flag"), py::arg("gen_python") = false);
    }

  }  // namespace PythonBindings
}  // namespace tiramisu
