#include "PyBuffer.h"

namespace tiramisu {
  namespace PythonBindings {

    void define_buffer(py::module &m){
      auto buffer_class = py::class_<buffer>(m, "buffer")
        .def(py::init<>())
        .def(py::init([]( // Create new constructor that doesn't use the last 2 args to avoid having to add the default values in python
            std::string name, 
            std::vector<tiramisu::expr> dim_sizes,
            tiramisu::primitive_t type,
            tiramisu::argument_t argt 
            //tiramisu::function,// *fct,
            //std::string// corr
        ) {
            return new buffer(name, dim_sizes, type, argt);
        }));
    }

  }  // namespace PythonBindings
}  // namespace tiramisu
