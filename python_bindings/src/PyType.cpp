#include "PyType.h"

namespace tiramisu {
  namespace PythonBindings {

    void define_type(py::module &m){
      auto primitive_t_enum = py::enum_<tiramisu::primitive_t>(m, "primitive_t").value("p_uint8", p_uint8)
	.value("p_uint16", p_uint16)
	.value("p_uint32", p_uint32)
	.value("p_uint64", p_uint64)
	.value("p_int8", p_int8)
	.value("p_int16", p_int16)
	.value("p_int32", p_int32)
	.value("p_int64", p_int64)
	.value("p_float32", p_float32)
	.value("p_float64", p_float64)
	.value("p_boolean", p_boolean)
	.value("p_async", p_async)
	.value("p_wait_ptr", p_wait_ptr)
	.value("p_void_ptr", p_void_ptr)
	.value("p_none", p_none).export_values();
    }

  }  // namespace PythonBindings
}  // namespace tiramisu
