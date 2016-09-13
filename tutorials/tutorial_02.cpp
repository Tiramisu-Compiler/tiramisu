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
#include "halide_image_io.h"

/* Halide code.
Func blurxy(Func input, Func blur_y) {
	Func blur_x;
	Var x, y, xi, yi;

	// The algorithm - no storage or order
	blur_x(x, y) = (input(x-1, y) + input(x, y) + input(x+1, y))/3;
	blur_y(x, y) = (blur_x(x, y-1) + blur_x(x, y) + blur_x(x, y+1))/3;

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
	coli::expr *e1_access1 = coli::expr::make(coli::type::op::access, coli::expr::make("c_input"), {coli::expr::make(coli::type::op::sub, coli::expr::make("i"), coli::expr::make((int32_t) 1)), coli::expr::make("j")});
	coli::expr *e1_access2 = coli::expr::make(coli::type::op::access, coli::expr::make("c_input"), {coli::expr::make("i"), coli::expr::make("j")});
	coli::expr *e1_access3 = coli::expr::make(coli::type::op::access, coli::expr::make("c_input"), {coli::expr::make(coli::type::op::add, coli::expr::make("i"), coli::expr::make((int32_t) 1)), coli::expr::make("j")});
	coli::expr *e1_add1    = coli::expr::make(coli::type::op::add, e1_access1, e1_access2);
	coli::expr *e1_add2    = coli::expr::make(coli::type::op::add, e1_add1, e1_access3);
	coli::expr *e1         = coli::expr::make(coli::type::op::div, e1_add2, coli::expr::make((uint8_t) 3));

	coli::expr *e2_access1 = coli::expr::make(coli::type::op::access, coli::expr::make("c_blurx"), {coli::expr::make("i"), coli::expr::make(coli::type::op::sub, coli::expr::make("j"), coli::expr::make((int32_t) 1))});
	coli::expr *e2_access2 = coli::expr::make(coli::type::op::access, coli::expr::make("c_blurx"), {coli::expr::make("i"), coli::expr::make("j")});
	coli::expr *e2_access3 = coli::expr::make(coli::type::op::access, coli::expr::make("c_blurx"), {coli::expr::make("i"), coli::expr::make(coli::type::op::add, coli::expr::make("j"), coli::expr::make((int32_t) 1))});
	coli::expr *e2_add1    = coli::expr::make(coli::type::op::add, e2_access1, e2_access2);
	coli::expr *e2_add2    = coli::expr::make(coli::type::op::add, e2_add1,    e2_access3);
	coli::expr *e2         = coli::expr::make(coli::type::op::div, e2_add2,    coli::expr::make((uint8_t) 3));

	coli::computation c_input("[N]->{c_input[i,j]: 0<=i<N and 0<=j<N}", NULL, false, &blurxy);
	coli::computation c_blurx("[N]->{c_blurx[i,j]: 0<i<N and 0<j<N}", e1,   true,  &blurxy);
	coli::computation c_blury("[N]->{c_blury[i,j]: 1<i<N-1 and 1<j<N-1}", e2,   true,  &blurxy);

	// Create a memory buffer (2 dimensional).
	coli::buffer b_blurx("b_blurx", 2, {SIZE,SIZE}, coli::type::primitive::uint8, NULL, false, coli::type::argument::none, &blurxy);

	// Map the computations to a buffer.
	c_input.set_access("{c_input[i,j]->b_input[i,j]}");
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

	blurxy.dump(true);

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
