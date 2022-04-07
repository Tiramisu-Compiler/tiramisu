#include "PyVar.h"

namespace tiramisu {
  namespace PythonBindings {

    void define_input(py::module &m){
      auto input_class = py::class_<input, computation>(m, "input")
        .def(py::init<std::string, std::vector<var>, tiramisu::primitive_t>())
        .def(py::init<std::vector<var>, tiramisu::primitive_t>())
        .def(py::init<std::string, std::vector<std::string>, std::vector<tiramisu::expr>, primitive_t>())
        .def("store_in", py::overload_cast<tiramisu::buffer*>(&computation::store_in))
        .def("store_in", py::overload_cast<tiramisu::buffer*, std::vector<tiramisu::expr>>(&computation::store_in));

    }

  }  // namespace PythonBindings
}  // namespace tiramisu