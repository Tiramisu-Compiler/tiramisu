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


/* Halide code.

Func blur_3x3(Func input, Func blur_y) {
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

The following code is the equivalent coli code.
It shows the high level picture, the exact code is below.

coli::function blur_3x3(
		coli::computation input,
		coli::computation blur_y)
   {
   	// Define the computations
	coli::computation blur_x: input(x, y) + input(x, y);
	blur_y: blur_x(x, y) + blur_x(x, y);

	// Set schedule
	blur_y.tile(0, 1, 256, 32);
	blur_y.vectorize(2, 8);
	blur_y.tag_parallel_dimension(0);
	blur_x.compute_at(blur_y, 1); //This call is subject to change.
	blur_x.vectorize(1, 8);

	// Generate time-processor domain
	gen_time_processor_domain();

	// Data mapping
	- Declare buffers
	- Bind the blur_3x3 arguments (which are computations) to their
	corresponding buffers.  This is a one-to-one mapping between buffers
	and computations.
	- Specify the data mapping between of blur_x
	
	// Gerate code.
  }
 */


int main(int argc, char **argv)
{

	// ---------------------------------------------------------------
	// Part I: define the computation without any schedule
	// and without any memory mapping.
	// ---------------------------------------------------------------

	// Declare a function.
	// This function is not a part of any library.
	// It represents the blur_3x3 pipeline in the above Halide code.
	coli::function func("blur_3x3", NULL);

	// Declare parameters that represent the image size.
	// The parameters will be used as bounds to the iteration space (i.e.
	// as loop bounds in the generated code).
	// Currently a parameter is simply a Halide expression, but in the
	// future it will be a coli::expression (i.e. an expression of
	// constants, symbolic constants, and computations).
	coli::parameter s1("s1", Halide::Expr((uint8_t) 100)); 
	coli::parameter s1("s2", Halide::Expr((uint8_t) 100));

	// Declare the computations of the function fct.
	// The first computation called "input" is simply a binding to the "input"
	// buffer that is passed to the function blur_3x3.
	// A binding is a 1 to 1 mapping between a computation and a buffer.
	// It is a way to represent a buffer as a computation so that our
	// coli::expressions are expression of coli::computations only rather
	// than being expressions of computations and buffers.
	coli::computation  c_input("[s1,s2]->{ c_input[i,j]: 0=<i<=s1 and 0<=j<=s2}", &fct);
	coli::computation c_blur_x("[s1,s2]->{c_blur_x[i,j]: 0<=i<=s1 and 0<=j<=s2}", &fct);
	coli::computation c_blur_y("[s1,s2]->{c_blur_y[i,j]: 0<=i<=s1 and 0<=j<=s2}", &fct);

	// Set the expression of c_blur_x and c_blur_y
	c_blur_x.set_expression(an isl_ast_expr * that represents the expression);
	c_blur_y.set_expression(an isl_ast_expr * that represents the expression);

	// Add the buffers as arguments to the function fct.
	fct.add_argument(c_input);
	fct.add_argument(c_blur_y);


	// ---------------------------------------------------------------
	// Part II: define the schedule of the computation.
	// ---------------------------------------------------------------

	
	// Set the schedule of each computation.
	c_blur_y.tile(0, 1, 256, 32);
	c_blur_y.vectorize(2, 8);
	c_blur_y.tag_parallel_dimension(0);
	c_blur_x.compute_at(c_blur_y, 1); //This call is subject to change.
	c_blur_x.vectorize(1, 8);


	// --------------------------------------------------------------
	// Part III: Generate the time-processors representation
	// which maps each computation to a time and a processor.
	// --------------------------------------------------------------

	// Generate the time-processor domain of each computation.
	func.gen_time_processor_domain();
	
	// --------------------------------------------------------------
	// Part IV: Create the buffers and do the mapping from computations
	// to buffers.
	// --------------------------------------------------------------

	// Create the 1st arguments of the function.
	coli::buffer buf0("b_input", 2, {s1,s2}, Halide::Int(8), NULL, &fct);
	// Create the 2nd arguments of the function.
	coli::buffer buf1("b_blury_y", 2, {s1,s2}, Halide::Int(8), NULL, &fct);

	// Bind the computations c_input and c_blur_y to their buffers.
	// This means that there is a one to one mapping between the elements
	// of the buffer and the elements of the computation.
	c_input.bind(b_input);
	c_blur_y.bind(b_blury_y);

	// Map the computation c_blur_x to a buffer
	// In this particular case we have also a one-to-one mapping but
	// it can be anything else. whereas a binding can only be
	// a one-to-one mapping.
	c_blur_x.SetWriteAccess("{c_blur_x[i,j]->b_blur_x[i, j]}");

	// --------------------------------------------------------------
	// Part V: Generate an ISL AST.
	// --------------------------------------------------------------

	// Generate an AST (abstract Syntax Tree)
	func.gen_isl_ast();

	// --------------------------------------------------------------
	// Part VI: Generate a Halide Stmt and an obj file representing
	// the whole function.
	// --------------------------------------------------------------

	// Generate Halide statement for the function.
	func.gen_halide_stmt();

	// Generate an object file from the function. 
	func.gen_halide_obj("build/generated_lib_tutorial_01.o");

	return 0;
}
