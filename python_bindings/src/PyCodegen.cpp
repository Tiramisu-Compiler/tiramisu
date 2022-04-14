#include "PyInit.h"
#include "../../include/tiramisu/core.h"
#include <pybind11/embed.h> // everything needed for embedding
#include <experimental/filesystem>

#define TO_STR2(x) #x
#define TO_STR(x) TO_STR2(x)

//std::stringstream ss;
//ss << TO_STR(HALIDE_BUILD);
std::string halide_build = TO_STR(HALIDE_BUILD);
//ss >> halide_build; // Extract into the string.
//#define str(s) #s
//#define HLB ""#HALIDE_BUILD""
//std::string halide_build = HLB;
//std::string halide_build = std::filesystem::path(halide_build_pre).parent_path;

namespace tiramisu {
  namespace PythonBindings {

    void define_codegen(py::module &m){
      m.def("codegen", 
            py::overload_cast<const std::vector<tiramisu::buffer *> &, const std::string, const bool, bool>(&tiramisu::codegen),
            "This function generates the declared function and computations in an object file",
            py::arg("arguments"), py::arg("obj_filename"), py::arg("gen_cuda_stmt") = false, py::arg("gen_python") = false);
      
      m.def("codegen", 
            py::overload_cast<const std::vector<tiramisu::buffer *> &, const std::string, const tiramisu::hardware_architecture_t, bool>(&tiramisu::codegen),
            "This function generates the declared function and computations in an object file",
            py::arg("arguments"), py::arg("obj_filename"), py::arg("gen_architecture_flag"), py::arg("gen_python") = false);

      m.def("pycodegen", [](const std::vector<tiramisu::buffer *> & buffs, const std::string name, const bool cuda)
	     -> void{
	       tiramisu::codegen(buffs, name, cuda, true);
	       function *fct = global::get_implicit_function();
	       std::string fname = fct->get_name();
	       //	       py::scoped_interpreter guard{};
	       
	       using namespace py::literals;
	       auto locals = py::dict("hbuild"_a = halide_build, "filename"_a = name, "funcname"_a = fname);
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

      m.def("pycodegen", [](const std::vector<tiramisu::buffer *> & buffs, const std::string name, const tiramisu::hardware_architecture_t arch, const bool cuda)
	     -> void{
	      tiramisu::codegen(buffs, name, arch, true);
	       function *fct = global::get_implicit_function();
	       std::string fname = fct->get_name();
	       //	       py::scoped_interpreter guard{};
	       
	       using namespace py::literals;
	       auto locals = py::dict("hbuild"_a = halide_build, "filename"_a = name, "funcname"_a = fname);
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
    }

  }  // namespace PythonBindings
}  // namespace tiramisu
