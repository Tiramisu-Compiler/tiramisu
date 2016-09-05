#include <isl/set.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>

#include <coli/debug.h>
#include <coli/core.h>

#include <String.h>
#include <Halide.h>


int main(int argc, char **argv)
{
	// Declare a function.
	coli::function fct("function0");

	// Declare the computations of the function fct.
	// To declare a computation, you need to provide:
	// (1) a Halide expression that represents the computation,
	// (2) an isl set representing the iteration space of the computation, and
	// (3) an isl context (which will be used by the ISL library calls).
	coli::computation computation0(Halide::Expr((uint8_t) 3), "{S0[i,j]: 0<=i<=1000 and 0<=j<=1000}", &fct);
	coli::computation computation1(Halide::Expr((uint8_t) 5), "{S1[i,j]: 0<=i<=1023 and 0<=j<=1023}", &fct);
	coli::computation computation2(Halide::Expr((uint8_t) 7), "{S2[i,j]: 0<=i<=1023 and 0<=j<=1023}", &fct);

	// Create a memory buffer (2 dimensional).
	coli::buffer buf0("buf0", 2, {10,10}, Halide::Int(8), NULL, &fct);

	// Add the buffer as an argument to the function fct.
	fct.add_argument(buf0);

	// Map the computations to the buffers (i.e. where each computation
	// should be stored in the buffer).
	computation0.SetWriteAccess("{S0[i,j]->buf0[i, j]}");
	computation1.SetWriteAccess("{S1[i,j]->buf0[0, 0]}");
	computation2.SetWriteAccess("{S2[i,j]->buf0[i, j]}");

	// Set the schedule of each computation.
	computation0.tile(0,1,32,32);
	computation1.set_schedule("{S1[i,j]->[2,i1,j1,i2,j3,j4]: i1=floor(i/32) and j1=floor(j/32) and i2=i and j3=floor(j/4) and j4=j%4 and 0<=i<=1023 and 0<=j<=1023}");
	computation2.split(0, 32);
	computation2.split(2, 32);
	computation2.interchange(1, 2);
	computation0.tag_parallel_dimension(1);
//	computation1.tag_vector_dimension(5);

	// Generate an AST (abstract Syntax Tree)
	func.gen_isl_ast();

	// Generate a Halide statement representing the function.
	func.gen_halide_stmt();

	// Dump IRs
	func.dump_iteration_space_IR();
	func.dump_schedule();
	func.dump_time_processor_IR();
	func.dump_halide_IR();

	// Generate an object file. 
	func.gen_halide_obj("build/generated_lib_tutorial_01.o");

	return 0;
}
