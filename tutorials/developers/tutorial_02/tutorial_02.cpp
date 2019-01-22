/**
  The goal of this tutorial is to implement in Tiramisu a code that is
  equivalent to the following

  for (int i = 0; i < 10; i++)
    for (int j = 0; j < 20; j++)
      output[i, j] = A[i, j] + i + 4;
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
    constant N("N", NN);
    constant M("M", MM);

    // Declare an input.  The input is declared by providing a name for the
    // input, the names of the input dimensions ("i" and "j" in this example),
    // the size of the input (which is NxM in this example), and the type of
    // the input elements.
    input A("A", {"i", "j"}, {N, M}, p_uint8);

    // Declare iterator variables.
    var i("i", 0, N), j("j", 0, M);

    // Declare the output computation.
    computation output("output", {i,j}, (A(i, j) + cast(p_uint8, i) + (uint8_t)4));

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
    buffer b_A("b_A", {expr(NN), expr(MM)}, p_uint8, a_input);
    buffer b_output("b_output", {expr(NN), expr(MM)}, p_uint8, a_output);

    // Map the computations to a buffer.
    // The following call indicates that each computation A[i,j]
    // is stored in the buffer element b_A[i,j] (one-to-one mapping).
    // This is the most common mapping to memory.
    A.store_in(&b_A);
    output.store_in(&b_output);

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    // Set the arguments and generate code
    tiramisu::codegen({&b_A, &b_output}, "build/generated_fct_developers_tutorial_02.o");

    return 0;
}
