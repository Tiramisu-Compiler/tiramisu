#include "PyVar.h"

namespace tiramisu {
  namespace PythonBindings {
 
    void define_input(py::module &m){
      auto input_class = py::class_<input, computation>(m, "input")
        .def(py::init<std::string, std::vector<var>, tiramisu::primitive_t>(), py::return_value_policy::reference)
        .def(py::init<std::vector<var>, tiramisu::primitive_t>(), py::return_value_policy::reference)
        .def(py::init<std::string, std::vector<std::string>, std::vector<tiramisu::expr>, primitive_t>(), py::return_value_policy::reference)
        .def(py::init([]( // Create new constructor that doesn't use the last 2 args to avoid having to add the default values in python
            std::string name, 
            std::vector<tiramisu::expr> dim_sizes,
            tiramisu::primitive_t type
            //tiramisu::function,// *fct,
            //std::string// corr
        ) {
            return new Input(name, dim_sizes, type);
        }), py::return_value_policy::reference)
        .def("get_buffer", &input::get_buffer, py::return_value_policy::reference)
        .def("get_name", &input::get_name, py::return_value_policy::reference)
        .def("store_in", py::overload_cast<tiramisu::buffer*>(&computation::store_in), py::keep_alive<1, 2>())
        .def("store_in", py::overload_cast<tiramisu::buffer*, std::vector<tiramisu::expr>>(&computation::store_in), py::keep_alive<1, 2>());

    }

  }  // namespace PythonBindings
}  // namespace tiramisu
