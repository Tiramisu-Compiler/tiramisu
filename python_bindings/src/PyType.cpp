#include "PyType.h"

namespace tiramisu {
  namespace PythonBindings {



    void define_type(py::module &m){
      auto expr_t_enum = py::enum_<tiramisu::expr_t>(m, "expr_t")
	.value("e_val", e_val)
	.value("e_var", e_var)
	.value("e_sync", e_sync)
	.value("e_op", e_op)
	.value("e_none", e_none)
	.export_values();
      
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

      py::implicitly_convertible<int, tiramisu::primitive_t>();

      auto op_t_enum = py::enum_<tiramisu::op_t>(m, "op_t")
	.value("o_minus", o_minus)
	.value("o_floor", o_floor)
	.value("o_sin", o_sin)
	.value("o_cos", o_cos)
	.value("o_tan", o_tan)
	.value("o_atan", o_atan)
	.value("o_asin", o_asin)
	.value("o_acos", o_acos)
	.value("o_sinh", o_sinh)
	.value("o_cosh", o_cosh)
	.value("o_tanh", o_tanh)
	.value("o_asinh", o_asinh)
	.value("o_acosh", o_acosh)
	.value("o_atanh", o_atanh)
	.value("o_abs", o_abs)
	.value("o_sqrt", o_sqrt)
	.value("o_expo", o_expo)
	.value("o_log", o_log)
	.value("o_ceil", o_ceil)
	.value("o_round", o_round)
	.value("o_trunc", o_trunc)
	.value("o_allocate", o_allocate)
	.value("o_free", o_free)
	.value("o_cast", o_cast)
	.value("o_address", o_address)
	.value("o_add", o_add)
	.value("o_sub", o_sub)
	.value("o_mul", o_mul)
	.value("o_div", o_div)
	.value("o_mod", o_mod)
	.value("o_logical_and", o_logical_and)
	.value("o_logical_or", o_logical_or)
	.value("o_logical_not", o_logical_not)
	.value("o_eq", o_eq)
	.value("o_ne", o_ne)
	.value("o_le", o_le)
	.value("o_lt", o_lt)
	.value("o_ge", o_ge)
	.value("o_gt", o_gt)
	.value("o_max", o_max)
	.value("o_min", o_min)
	.value("o_right_shift", o_right_shift)
	.value("o_left_shift", o_left_shift)
	.value("o_memcpy", o_memcpy)
	.value("o_select", o_select)
	.value("o_cond", o_cond)
	.value("o_lerp", o_lerp)
	.value("o_call", o_call)
	.value("o_access", o_access)
	.value("o_address_of", o_address_of)
	.value("o_select", o_select)
	.value("o_cond", o_cond)
	.value("o_lerp", o_lerp)
	.value("o_call", o_call)
	.value("o_access", o_access)
	.value("o_address_of", o_address_of)
	.value("o_lin_index", o_lin_index)
	.value("o_type", o_type)
	.value("o_dummy", o_dummy)
	.value("o_buffer", o_buffer)
	.value("o_none", o_none).export_values();

      auto argument_t_enum = py::enum_<tiramisu::argument_t>(m, "argument_t")
	.value("a_input", a_input)
	.value("a_output", a_output)
	.value("a_temporary", a_temporary)
	.export_values();

      auto rank_t_enum = py::enum_<tiramisu::rank_t>(m, "rank_t")
	.value("r_sender", tiramisu::rank_t::r_sender)
	.value("r_receiver", tiramisu::rank_t::r_receiver).export_values();

      auto hardware_architecture_t_enum = py::enum_<tiramisu::hardware_architecture_t>(m, "hardware_architecture_t")
	.value("arch_cpu", tiramisu::hardware_architecture_t::arch_cpu)
	.value("arch_nvidia_gpu", tiramisu::hardware_architecture_t::arch_nvidia_gpu)
	.value("arch_flexnlp", tiramisu::hardware_architecture_t::arch_flexnlp).export_values();

    }



  }  // namespace PythonBindings
}  // namespace tiramisu
