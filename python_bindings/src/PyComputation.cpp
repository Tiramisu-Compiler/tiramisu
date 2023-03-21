#include "PyConstant.h"

namespace tiramisu {
  namespace PythonBindings {

    void define_computation(py::module &m){
    
      auto computation_class = py::class_<computation>(m, "computation")
        .def(py::init<std::string, std::vector< var >, tiramisu::expr >(),
	     py::return_value_policy::reference, py::keep_alive<0, 1>(), py::keep_alive<0, 2>())
        .def(py::init<std::string, std::vector< var >, tiramisu::primitive_t >(),
	     py::return_value_policy::reference, py::keep_alive<0, 1>())
        .def(py::init<std::vector< var >, tiramisu::primitive_t >(),
	     py::return_value_policy::reference, py::keep_alive<0, 1>())
        // .def("__call__", &computation::operator())
        // .def("__call__", py::overload_cast<expr, expr>(&computation::operator())) // temporary workaround
        .def("__getitem__", [](tiramisu::computation &c, std::vector<expr> a) -> expr {
            switch (a.size()) {
                case 1: return c.template operator()<expr>(a[0]);
                case 2: return c.template operator()<expr, expr>(a[0], a[1]);
                case 3: return c.template operator()<expr, expr, expr>(a[0], a[1], a[2]);
                case 4: return c.template operator()<expr, expr, expr, expr>(a[0], a[1], a[2], a[3]);
                case 5: return c.template operator()<expr, expr, expr, expr, expr>(a[0], a[1], a[2], a[3], a[4]);
                case 6: return c.template operator()<expr, expr, expr, expr, expr, expr>(a[0], a[1], a[2], a[3], a[4], a[5]);
                case 7: return c.template operator()<expr, expr, expr, expr, expr, expr, expr>(a[0], a[1], a[2], a[3], a[4], a[5], a[6]);
                case 8: return c.template operator()<expr, expr, expr, expr, expr, expr, expr, expr>(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
                case 9: return c.template operator()<expr, expr, expr, expr, expr, expr, expr, expr, expr>(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]);
                case 10: return c.template operator()<expr, expr, expr, expr, expr, expr, expr, expr, expr, expr>(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9]);
                case 11: return c.template operator()<expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr>(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10]);
                case 12: return c.template operator()<expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr>(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11]);
                case 13: return c.template operator()<expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr>(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12]);
                case 14: return c.template operator()<expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr>(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13]);
                case 15: return c.template operator()<expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr>(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14]);
                case 16: return c.template operator()<expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr>(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15]);
                case 17: return c.template operator()<expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr>(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16]);
                case 18: return c.template operator()<expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr>(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17]);
                case 19: return c.template operator()<expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr>(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18]);
                case 20: return c.template operator()<expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr>(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19]);
                case 21: return c.template operator()<expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr>(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20]);
                case 22: return c.template operator()<expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr>(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21]);
                case 23: return c.template operator()<expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr>(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22]);
                case 24: return c.template operator()<expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr>(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23]);
                case 25: return c.template operator()<expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr>(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24]);
                case 26: return c.template operator()<expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr>(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25]);
                case 27: return c.template operator()<expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr>(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26]);
                case 28: return c.template operator()<expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr>(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27]);
                case 29: return c.template operator()<expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr>(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28]);
                case 30: return c.template operator()<expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr>(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29]);
                case 31: return c.template operator()<expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr>(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30]);
                case 32: return c.template operator()<expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr, expr>(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31]);
            }
            throw std::invalid_argument("invalid number of arguments");
	  }, py::return_value_policy::reference, py::keep_alive<0, 1>(), py::keep_alive<0, 2>())
        .def("parallelize", &computation::parallelize)
        .def("store_in", py::overload_cast<tiramisu::buffer*>(&computation::store_in),
	     py::keep_alive<1, 2>())
        .def("store_in", py::overload_cast<tiramisu::buffer*, std::vector<tiramisu::expr>>(&computation::store_in),
	     py::keep_alive<1, 2>())
        .def("tile", py::overload_cast<var, var, int, int>(&computation::tile),
	     py::keep_alive<1, 2>(), py::keep_alive<1, 3>())
        .def("tile", py::overload_cast<var, var, int, int, var, var, var, var>(&computation::tile))
        .def("tile", py::overload_cast<var, var, var, int, int, int>(&computation::tile))
        .def("tile", py::overload_cast<var, var, var, int, int, int, var, var, var, var, var, var>(&computation::tile))
        .def("tile", py::overload_cast<int, int, int, int>(&computation::tile))
        .def("tile", py::overload_cast<int, int, int, int, int, int>(&computation::tile))
        .def("after", py::overload_cast<computation&, var>(&computation::after), py::keep_alive<1, 2>())
        .def("after", py::overload_cast<computation&, int>(&computation::after), py::keep_alive<1, 2>())
        .def("set_expression", &computation::set_expression, py::keep_alive<1, 2>())
        .def("gpu_tile", py::overload_cast<var, var, int, int>(&computation::gpu_tile))
        .def("gpu_tile", py::overload_cast<var, var, int, int, var, var, var, var>(&computation::gpu_tile))
        .def("gpu_tile", py::overload_cast<var, var, var, int, int, int>(&computation::gpu_tile))
        .def("gpu_tile", py::overload_cast<var, var, var, int, int, int, var, var, var, var, var, var>(&computation::gpu_tile))
        .def("then", py::overload_cast<computation&, var>(&computation::then), py::keep_alive<1, 2>())
	.def("then", py::overload_cast<computation&, int>(&computation::then), py::keep_alive<1, 2>())
        .def("split", py::overload_cast<var, int>(&computation::split))
        .def("split", py::overload_cast<var, int, var, var>(&computation::split))
        .def("split", py::overload_cast<int, int>(&computation::split))
        .def("cache_shared", &computation::cache_shared)
        .def("get_buffer", &computation::get_buffer, py::return_value_policy::reference, py::keep_alive<0, 1>())
        .def("after_low_level", py::overload_cast<computation&, int>(&computation::after_low_level))
        .def("dump", &computation::dump);
    }

  }  // namespace PythonBindings
}  // namespace tiramisu
