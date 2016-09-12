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

/* Halide code.
Func blurxy(Func input, Func blur_y) {
	Func blur_x;
	Var x, y, xi, yi;

	// The algorithm - no storage or order
	blur_x(x, y) = input(x, y) + input(x, y);
	blur_y(x, y) = blur_x(x, y) + blur_x(x, y);

	// The schedule - defines order, locality; implies storage
	blur_y.tile(x, y, xi, yi, 256, 32)
		.vectorize(xi, 8).parallel(y);
	blur_x.compute_at(blur_y, x).vectorize(x, 8);
  }
*/

#define SIZE 10

int main(int argc, char **argv)
{
	// Set default coli options.
	coli::global::set_default_coli_options();

	/*
	 * Declare a function blurxy.
	 * Declare two arguments (coli buffers) for the function: b_input and b_blury
	 * Declare an invariant for the function.
	 */
	coli::function blurxy("blurxy");
	coli::buffer b_input("b_input", 2, {SIZE,SIZE}, coli::type::primitive::uint8, NULL, true, coli::type::argument::input, &blurxy);
	coli::buffer b_blury("b_blury", 2, {SIZE,SIZE}, coli::type::primitive::uint8, NULL, true, coli::type::argument::output, &blurxy);
	coli::invariant p0("N", coli::expr::make((int32_t) SIZE), &blurxy);

	// Declare the computations c_blurx and c_blury.
	coli::expr *e1 = coli::expr::make(coli::type::op::add, coli::expr::make((uint8_t) 1), coli::expr::make((uint8_t) 1));
	coli::expr *e2 = coli::expr::make(coli::type::op::access, coli::expr::make("c_blurx"), {coli::expr::make((int32_t) 0), coli::expr::make((int32_t) 0)});

	coli::computation c_blurx("[N]->{c_blurx[i,j]: 0<=i<N and 0<=j<N}", e1, &blurxy);
	coli::computation c_blury("[N]->{c_blury[i,j]: 0<=i<N and 0<=j<N}", e2, &blurxy);

	// Create a memory buffer (2 dimensional).
	coli::buffer b_blurx("b_blurx", 2, {SIZE,SIZE}, coli::type::primitive::uint8, NULL, false, coli::type::argument::none, &blurxy);

	// Map the computations to a buffer.
	c_blurx.set_access("{c_blurx[i,j]->b_blurx[i,j]}");
	c_blury.set_access("{c_blury[i,j]->b_blury[i,j]}");

	// Set the schedule of each computation.
	// The identity schedule means that the program order is not modified
	// (i.e. no optimization is applied).
	c_blurx.tile(0,1,2,2);
	c_blurx.tag_parallel_dimension(0);
	c_blury.set_schedule("{c_blury[i,j]->[i,j]}");
	c_blury.after(c_blurx, coli::computation::root_dimension);

	// Set the arguments to blurxy
	blurxy.set_arguments({&b_input, &b_blury});

	// Generate code
	blurxy.gen_isl_ast();
	blurxy.gen_halide_stmt();
	blurxy.gen_halide_obj("build/generated_lib_tutorial_02.o");

	// Some debugging
	blurxy.dump_iteration_domain();
	blurxy.dump_halide_stmt();

	blurxy.dump(true);

	return 0;
}
