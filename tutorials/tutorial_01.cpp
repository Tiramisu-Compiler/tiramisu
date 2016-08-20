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

	// Create a memory buffer (2 dimensional).
	coli::buffer buf0("buf0", 2, {10,10}, Halide::Int(8), NULL, &fct);

	// Add the buffer as an argument to the function fct.
	fct.add_argument(buf0);

	// Map the computations to the buffers (i.e. where each computation
	// should be stored in the buffer).
	computation0.SetWriteAccess("{S0[i,j]->buf0[i, j]}");

	// Set the schedule of each computation.
	computation0.set_schedule("{S0[i,j]->[i,j]: 0<=i<10 and 0<=j<10}");
	computation0.tag_parallel_dimension(0);

	// Generate an AST (abstract Syntax Tree)
	lib.gen_isl_ast();

	// Generate Halide statement for each function in the library.
	lib.gen_halide_stmt();

	// If you want to get the generated halide statements, call
	// lib.get_halide_stmts().  This will return a vector of
	// Halide::Internal::Stmt*.  Each one among these statements
	// represents a function in the library.

	// Dump the iteration space IR (input) and the the Halide IR (output)
	// for each function in the library.
	lib.dump_iteration_space_IR();
	lib.dump_halide_IR();

	// Generate an object file from the library lib.
	lib.gen_halide_obj("build/generated_lib_tutorial_01.o");

	return 0;
}
