/**
  The goal of this tutorial is to implement, in Tiramisu, a code that is
  equivalent to the following

  for (int i = 0; i < 10; i++)
      buf0[i] = 3 + 4;

  Every Tiramisu program needs to include the header file tiramisu/tiramisu.h
  which defines classes for declaring and compiling Tiramisu expressions.

  Tiramisu is a code generator, therefore the goal of a Tiramisu program is to
  generate code.  The generated code is supposed to be called from another
  program (the user program).

  A Tiramisu program is structures as follows:
	- It starts with a call to initialize the Tiramisu compiler.  The name of the
	function to be generated is specified during this initialization.
	- The user then declares the algorithm: Tiramisu expressions.
	- The user then specifies how the algorithm should be optimized
	using scheduling and data mapping commands.
	- The user then calls the codegen() function which generates code.
	It compiles the Tiramisu program, generates the optimized program
	in an object file.

  The user can then call the function declared in the generated object file
  from any place in his program.
 *
 */

#include <tiramisu/tiramisu.h>

using namespace tiramisu;

int main(int argc, char **argv)
{
    // Set default tiramisu options.
    global::set_default_tiramisu_options();

    // Declare a function called "function0".
    // A function in tiramisu is the equivalent of a function in C.
    // It can have input and output arguments (buffers).  These arguments
    // are declared later in the tutorial.
    function function0("function0");


    // -------------------------------------------------------
    // Layer I: provide the algorithm.
    // -------------------------------------------------------

    // Declare an expression that will be associated to the
    // computations.  This expression sums 3 and 4.
    expr e = expr(3) + expr(4);

    // Declare a computation within function0.
    // To declare a computation, you need to provide:
    // (1) an ISL set representing the iteration space of the computation.
    // Tiramisu uses the ISL syntax to represent sets and maps.  The ISL syntax
    // is described in http://barvinok.gforge.inria.fr/barvinok.pdf (Section
    // 1.2.1 for sets and iteration domains, and 1.2.2 for maps and access
    // relations),
    // (2) a tiramisu expression: this is the expression that will be computed
    // by the computation.
    // (3) the type of the computation which is an unsigned char in this case (p_uint8).
    // (4) a boolean that indicates whether this computation should be
    // scheduled or not (more about this in the next tutorials).
    // (5) the function in which the computation will be declared.
    //
    // The best way to learn about the constructor of computations is to
    // check the documentation of the computation class in
    // https://tiramisu-compiler.github.io/doc
    computation S0("{S0[i]: 0<=i<10}", e, true, p_uint8, &function0);

    // ------------------------------------------------------------
    // Layer II: specify how to schedule (optimize) the algorithm.
    // ------------------------------------------------------------

    // Declare an iterator variable.
    var i("i");

    // Set the loop level i (i.e., the loop level that uses i as iterator).
    S0.parallelize(i);


    // -------------------------------------------------------
    // Layer III: allocate buffers and specify how computations
    // should be stored in these buffers.
    // -------------------------------------------------------

    // Create a buffer buf0.
    buffer buf0("buf0", {expr(10)}, p_uint8, a_output, &function0);

    // Map the computation S0 to the buffer buf0.
    // This means specifying where each computation S0(i) should be
    // stored exactly in the buffer buf0.
    // The data mapping is specified using an ISL map.  The mapping
    // used in this example indicates that S0[i] should be stored in the
    // buffer location buf0[i].
    // Tiramisu uses the ISL syntax to represent sets and maps.  The ISL syntax
    // is described in http://barvinok.gforge.inria.fr/barvinok.pdf (Section
    // 1.2.1 for sets and iteration domains, and 1.2.2 for maps and access
    // relations).
    S0.set_access("{S0[i]->buf0[i]}");



    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    // Set buf0 as an argument to the function.
    // The buffer buf0 is supposed to be allocated  by the user (caller)
    // and passed to the generated function "function0".
    // Any buffer of type a_output or a_input are supposed to be allocated
    // by the caller, in contrast to buffers of type a_temporary which are
    // allocated automatically by the Tiramisu runtime within the callee
    // and should not be passed as arguments to the function).
    function0.set_arguments({&buf0});

    // Generate the time-processor domain of the computation.
    function0.gen_time_space_domain();

    // Generate an AST (abstract Syntax Tree)
    function0.gen_isl_ast();

    // Generate Halide statement for the function.
    function0.gen_halide_stmt();

    // Generate an object file from the function.
    function0.gen_halide_obj("build/generated_fct_developers_tutorial_01.o");

    return 0;
}

/**
 * Remarques
 * ----------
 * - Note that the name used during the construction of a tiramisu object and the
 *   identifier of that object are identical (for example buf0, "buf0").
 *   This is not required but highly recommended as it simplifies reading
 *   tiramisu code and prevents errors.
 */
