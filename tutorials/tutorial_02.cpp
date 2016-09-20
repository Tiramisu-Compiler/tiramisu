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

#define SIZE0 1280
#define SIZE1 768

using namespace coli;

int main(int argc, char **argv)
{
	// Set default coli options.
	global::set_default_coli_options();

	/*
	 * Declare a function blurxy.
	 * Declare two arguments (coli buffers) for the function: b_input and b_blury
	 * Declare an invariant for the function.
	 */
	function blurxy("blurxy");
	buffer b_input("b_input", 2, {SIZE0,SIZE1}, p_uint8, NULL, a_input, &blurxy);
	buffer b_blury("b_blury", 2, {SIZE0,SIZE1}, p_uint8, NULL, a_output, &blurxy);
	invariant p0("N", expr((int32_t) SIZE0), &blurxy);
	invariant p1("M", expr((int32_t) SIZE1), &blurxy);

	// Declare the computations c_blurx and c_blury.
	computation c_input("[N]->{c_input[i,j]: 0<=i<N and 0<=j<N}", NULL, false, p_uint8, &blurxy);

        expr e1 = (c_input(idx("i") - 1, idx("j")) +
                   c_input(idx("i")    , idx("j")) +
                   c_input(idx("i") + 1, idx("j")))/((uint8_t) 3);

	computation c_blurx("[N,M]->{c_blurx[i,j]: 0<i<N and 0<j<M}", &e1, true, p_uint8, &blurxy);

	expr e2 = (c_blurx(idx("i"), idx("j") - 1) +
	           c_blurx(idx("i"), idx("j")) +
	           c_blurx(idx("i"), idx("j") + 1))/((uint8_t) 3);

	computation c_blury("[N,M]->{c_blury[i,j]: 1<i<N-1 and 1<j<M-1}", &e2, true, p_uint8, &blurxy);

	// Create a memory buffer (2 dimensional).
	buffer b_blurx("b_blurx", 2, {SIZE0,SIZE1}, p_uint8, NULL, a_temporary, &blurxy);

	// Map the computations to a buffer.
	c_input.set_access("{c_input[i,j]->b_input[i,j]}");
	c_blurx.set_access("{c_blurx[i,j]->b_blurx[i,j]}");
	c_blury.set_access("{c_blury[i,j]->b_blury[i,j]}");

	// Set the schedule of each computation.
	// The identity schedule means that the program order is not modified
	// (i.e. no optimization is applied).
	c_blurx.tile(0,1,2,2);
	c_blurx.tag_gpu_dimensions(0,1);
	c_blury.set_schedule("{c_blury[i,j]->[i,j]}");
	c_blury.after(c_blurx, computation::root_dimension);

	// Set the arguments to blurxy
	blurxy.set_arguments({&b_input, &b_blury});

	// Generate code
	blurxy.gen_isl_ast();
	blurxy.gen_halide_stmt();
	blurxy.gen_halide_obj("build/generated_fct_tutorial_02.o");

	// Some debugging
	blurxy.dump_iteration_domain();
	blurxy.dump_halide_stmt();

	// Dump all the fields of the blurxy class.
	blurxy.dump(true);

	return 0;
}
