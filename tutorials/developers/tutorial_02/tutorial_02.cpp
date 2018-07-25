/**
  The goal of this tutorial is to implement in Tiramisu a code that is
  equivalent to the following

  for (int i = 0; i < 10; i++)
    for (int j = 0; j < 20; j++)
      output[i, j] = input[i, j] + i + 2;
*/

#include <tiramisu/tiramisu.h>

#define NN 10
#define MM 20

using namespace tiramisu;

int main(int argc, char **argv)
{
    tiramisu::init("tut_02");

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    // Declare two constants N and M. These constants will be used as loop bounds.
    constant N_const("N", NN);
    constant M_const("M", MM);

    // Declare iterator variables.
    var i("i", 0, N_const), j("j", 0, M_const);

    // Declare a wrapper around the input.
    // In Tiramisu, if a function reads an input buffer (or writes to it), that buffer
    // is not accessed directly, but should first be wrapped in a computation.
    // This is mainly because computations in Tiramisu do not access memory directly,
    // since the algorithm is supposed to be expressed independently of how data is stored.
    // Therefore computations (algorithms) access only other computations.  The actual data
    // layout is only specified later in Layer III.
    // A wrapper is usually declared by providing the iterators (that define the size of the
    // buffer) and the type of the buffer elements.
    computation input({i,j}, p_uint8);

    // Declare the output computation.
    computation output({i,j}, (input(i, j) + cast(p_uint8, i) + (uint8_t)4));

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    // Set the schedule of the computation.
    var i0("i0"), i1("i1"), j0("j0"), j1("j1");
    
    // Tile the i, j loop around output by a 2x2 tile. The names of iterators
    // in the resulting loop are i0, j0, i1, j1.
    output.tile(i, j, 2, 2, i0, j0, i1, j1);
  
    // Parallelize the outermost loop i0 (OpenMP style parallelism).
    output.parallelize(i0);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    // Declare input and output buffers.
    buffer b_input("b_input", {expr(NN), expr(MM)}, p_uint8, a_input);
    buffer b_output("b_output", {expr(NN), expr(MM)}, p_uint8, a_output);

    // Map the computations to a buffer.
    // The following call indicates that each computation input[i,j]
    // is stored in the buffer element b_input[i,j] (one-to-one mapping).
    // This is the most common mapping to memory.
    input.store_in(&b_input);
    output.store_in(&b_output);

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    // Set the arguments and generate code
    tiramisu::codegen({&b_input, &b_output}, "build/generated_fct_developers_tutorial_02.o");

    return 0;
}
