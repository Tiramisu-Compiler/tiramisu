#include <isl/set.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>

#include <coli/debug.h>
#include <coli/core.h>

#include <string.h>
#include <Halide.h>


int main(int argc, char **argv)
{
	// Set default coli options.
	coli::context::set_default_coli_options();

	// Declare a library.  A library is composed of a set of functions.
	coli::library lib("library0");

	// Declare a function in the library lib.
	coli::function fct("function0", &lib);

	// Declare the computations of the function fct.
	// To declare a computation, you need to provide:
	// (1) a Halide expression that represents the computation,
	// (2) an isl set representing the iteration space of the computation, and
	// (3) an isl context (which will be used by the ISL library calls).
	coli::computation computation0(Halide::Expr((uint8_t) 3), "{S0[i,j]: 0<=i<10 and 0<=j<10}", &fct);
	coli::computation computation1(Halide::Expr((uint8_t) 3), "{S1[i,j]: 0<=i<10 and 0<=j<10}", &fct);

	// Create a memory buffer (2 dimensional).
	coli::buffer buf0("buf0", 2, {10,10}, Halide::Int(8), NULL, &fct);

	// Add the buffer as an argument to the function fct.
	fct.add_argument(buf0);

	// Map the computations to a buffer (i.e. where each computation
	// should be stored in the buffer).
	// This mapping will be updated automaticall when the schedule
	// is applied.  To disable automatic data mapping updates use
	// coli::context::set_auto_data_mapping(false).
	computation0.SetWriteAccess("{S0[i,j]->buf0[i,j]}");
	computation1.SetWriteAccess("{S1[i,j]->buf0[i,j]}");

	// Dump the iteration space IR (input)
	// for each function in the library.
	lib.dump_iteration_space_IR();

	// Set the schedule of each computation.
	// The identity schedule means that the program order is not modified
	// (i.e. no optimization is applied).
	computation0.tile(0,1,2,2);
	computation0.tag_parallel_dimension(0);
	computation1.set_schedule("{S1[i,j]->[i,j]: 0<=i<10 and 0<=j<10}");
	computation1.after(computation0, coli::root_dimension);

	// Generate the time-processor IR of each computation in the library
	// and dump the time-processor IR on stdout
	lib.gen_time_processor_IR();
	lib.dump_time_processor_IR();

	// Generate an AST (abstract Syntax Tree)
	lib.gen_isl_ast();

	// Generate Halide statement for each function in the library.
	lib.gen_halide_stmt();

	// If you want to get the generated halide statements, call
	// lib.get_halide_stmts().  This will return a vector of
	// Halide::Internal::Stmt*.  Each one among these statements
	// represents a function in the library.

	// Dump the Halide IR (output)
	// for each function in the library.
	lib.dump_halide_IR();

	// Generate an object file from the library lib.
	lib.gen_halide_obj("build/generated_lib_tutorial_01.o");

	return 0;
}
