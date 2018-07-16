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
    tiramisu::init();

    // Declare a function called "function0".
    // A function in tiramisu is the equivalent of a function in C.
    // It can have input and output arguments (buffers).  These arguments
    // are declared later in the tutorial.
    function function0("function0");


    // -------------------------------------------------------
    // Layer I: provide the algorithm.
    // -------------------------------------------------------

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
    computation S0("{S0[i]: 0<=i<10}", expr(3) + expr(4), true, p_uint8, &function0);

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
    S0.bind_to(&buf0);

    // Another way to specify the data mapping is use an ISL mapping, by calling
    // 		S0.set_access("{S0[i]->buf0[i]}");
    // This mapping indicates that S0[i] should be stored in the buffer location buf0[i].
    // Tiramisu uses the ISL syntax to represent sets and maps.  The ISL syntax
    // is described in http://barvinok.gforge.inria.fr/barvinok.pdf (Section
    // 1.2.1 for sets and iteration domains, and 1.2.2 for maps and access
    // relations).

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    // Generate code and compile it to an object file.  Two arguments need
    // to be passed to the code generator:
    //	    - The arguments (buffers) t passed to the generated function.
    //	      In this example, the buffer buf0 is set as an argument to the function.
    //	      The buffer buf0 is supposed to be allocated  by the user (caller)
    //	      and passed to the generated function "function0".
    //	      Any buffer of type a_output or a_input are supposed to be allocated
    //	      by the caller, in contrast to buffers of type a_temporary which are
    //	      allocated automatically by the Tiramisu runtime within the callee
    //	      and should not be passed as arguments to the function).
    //	    - The name of the object file to be generated.
    function0.codegen({&buf0}, "build/generated_fct_developers_tutorial_01");

    return 0;
}

/**
 * - Note that the name used during the construction of a tiramisu object and the
 *   identifier of that object are identical (for example buf0, "buf0").
 *   This is not required but highly recommended as it simplifies reading
 *   tiramisu code and prevents errors.
 */
