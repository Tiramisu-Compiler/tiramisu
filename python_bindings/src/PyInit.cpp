#include "PyInit.h"
#include "../../include/tiramisu/core.h"

namespace tiramisu
{
  namespace PythonBindings
  {

    void init_py(std::string name)
    {
      tiramisu::init(name);
    }

    void define_init(py::module &m)
    {
      m.def("init", &init_py, "Set up Tiramisu default function", py::return_value_policy::reference);
      m.def("get_implicit_function", &global::get_implicit_function, py::return_value_policy::reference);

      // Legality functions
      m.def("perform_full_dependency_analysis", &perform_full_dependency_analysis);
      m.def("prepare_schedules_for_legality_checks", &prepare_schedules_for_legality_checks, "Function that prepares the implicit function for legality check", py::arg("reset_static_dimesion") = false);
      m.def("check_legality_of_function", &check_legality_of_function);
      m.def("loop_unrolling_is_legal", py::overload_cast<var, std::vector<computation *>>(&loop_unrolling_is_legal));
      m.def("loop_unrolling_is_legal", py::overload_cast<int, std::vector<computation *>>(&loop_unrolling_is_legal));
      m.def("loop_parallelization_is_legal", py::overload_cast<var, std::vector<computation *>>(&loop_parallelization_is_legal));
      m.def("loop_parallelization_is_legal", py::overload_cast<int, std::vector<computation *>>(&loop_parallelization_is_legal));
    }

  } // namespace PythonBindings
} // namespace tiramisu
