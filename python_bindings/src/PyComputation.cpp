#include "PyConstant.h"

namespace tiramisu {
  namespace PythonBindings {

    void define_computation(py::module &m){
      auto const_class = py::class_<computation>(m, "computation")
        .def(py::init<std::string, std::vector< var >, tiramisu::expr >())
        .def("parallelize", &computation::parallelize)
        .def("store_in", py::overload_cast<tiramisu::buffer*>(&computation::store_in))
        .def("store_in", py::overload_cast<tiramisu::buffer*, std::vector<tiramisu::expr>>(&computation::store_in))
        .def("tile", py::overload_cast<var, var, int, int>(&computation::tile))
        .def("tile", py::overload_cast<var, var, int, int, var, var, var, var>(&computation::tile))
        .def("tile", py::overload_cast<var, var, var, int, int, int>(&computation::tile))
        .def("tile", py::overload_cast<var, var, var, int, int, int, var, var, var, var, var, var>(&computation::tile))
        .def("tile", py::overload_cast<int, int, int, int>(&computation::tile))
        .def("tile", py::overload_cast<int, int, int, int, int, int>(&computation::tile));
    }

  }  // namespace PythonBindings
}  // namespace tiramisu
