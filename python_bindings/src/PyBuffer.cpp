#include "PyBuffer.h"

namespace tiramisu {
  namespace PythonBindings {

    void define_buffer(py::module &m){
      auto buffer_class = py::class_<buffer>(m, "buffer")
        .def(py::init<>(), py::return_value_policy::reference)
        .def(py::init([]( // Create new constructor that doesn't use the last 2 args to avoid having to add the default values in python
            std::string name, 
            std::vector<tiramisu::expr> dim_sizes,
            tiramisu::primitive_t type,
            tiramisu::argument_t argt 
            //tiramisu::function,// *fct,
            //std::string// corr
        ) {
            return new buffer(name, dim_sizes, type, argt);
		      }), py::return_value_policy::reference, py::keep_alive<0, 2>())
        .def("get_name", &buffer::get_name)
        .def("dump", &buffer::dump);

      buffer_class.def("allocate_at", py::overload_cast<tiramisu::computation &, tiramisu::var>(&buffer::allocate_at), py::keep_alive<1, 1>());
    }

  }  // namespace PythonBindings
}  // namespace tiramisu
