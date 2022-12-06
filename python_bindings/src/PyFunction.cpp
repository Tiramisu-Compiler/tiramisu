#include "PyFunction.h"
#include <pybind11/embed.h> // everything needed for embedding
#include <experimental/filesystem>

#define TO_STR2(x) #x
#define TO_STR(x) TO_STR2(x)
std::string halide_build_p = TO_STR(HALIDE_BUILD);

namespace tiramisu {

  namespace PythonBindings {
    void define_function(py::module &m){
      auto function_class = py::class_<function>(m, "function")
	.def(py::init<std::string>(), py::return_value_policy::reference)
	.def("dump", &function::dump)
	.def("gen_c_code", &function::gen_c_code)
	.def("dump_halide_stmt", &function::dump_halide_stmt)
	.def("codegen", py::overload_cast<const std::vector<tiramisu::buffer *> &, const std::string, const bool, bool>(&tiramisu::function::codegen));

      function_class.def("pycodegen", [](tiramisu::function & fct, const std::vector<tiramisu::buffer *> & buffs, const std::string name, const bool cuda)
	     -> void{
			   fct.codegen(buffs, name, cuda, true);
	       std::string fname = fct.get_name();
	       //	       py::scoped_interpreter guard{};
	       
	       using namespace py::literals;
	       auto locals = py::dict("hbuild"_a = halide_build_p, "filename"_a = name, "funcname"_a = fname);
	       py::exec(R"(
from distutils.core import Extension
import os
import Cython
from pathlib import Path
from Cython.Build.Inline import _get_build_extension
from Cython.Build.Dependencies import cythonize
tmp = hbuild
dir = str(Path(tmp).parent)
extension = Extension(name=funcname, language='c++', sources=[filename + '.py.cpp'], extra_objects=[filename, dir + "/python_bindings/libHalide_PyStubs.a"], include_dirs=[dir + "/include"])
build_extension = _get_build_extension()
build_extension.extensions = cythonize([extension],
                                       include_path=[dir + "/include"], quiet=False)
build_extension.build_lib = os.path.dirname(filename)
build_extension.run()
)", py::globals(), locals);
		 }
	      );
      //Printer, add buffers, get buffers?

    }


  }  // namespace PythonBindings
}  // namespace tiramisu




