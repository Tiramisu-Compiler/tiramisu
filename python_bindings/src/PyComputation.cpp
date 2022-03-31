#include "PyConstant.h"

namespace tiramisu {
  namespace PythonBindings {

    void define_computation(py::module &m){
      auto const_class = py::class_<computation>(m, "computation")
        .def(py::init<std::string, std::vector< var >, tiramisu::expr >())
        .def("parallelize", &computation::parallelize)
        .def("store_in", py::overload_cast<tiramisu::buffer*>(&computation::store_in));
    }

  }  // namespace PythonBindings
}  // namespace tiramisu
