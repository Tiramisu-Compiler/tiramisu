/**
  copy : copies vector to another vector

  for (int i = 0; i < N; i++)
      output[i] = A[i];
*/

#include <tiramisu/tiramisu.h>

#define NN 10

using namespace tiramisu;

int main(int argc, char **argv)
{
    tiramisu::init("copy");

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------
    // Declare the constant N.
    constant N("N", NN);
    // Declare iterator variable.
    var i("i", 0, N);
    // Declare the input.
    input A("A", {i}, p_uint8);
    // Declare the output computation.
    computation output("output", {i}, A(i));

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    // Declare input and output buffers.
    buffer b_A("b_A", {expr(NN)}, p_uint8, a_input);
    buffer b_output("b_output", {expr(NN)}, p_uint8, a_output);
    // Map the computations to a buffer.
    A.store_in(&b_A);
    output.store_in(&b_output);

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    // Set the arguments and generate code
    tiramisu::codegen({&b_A, &b_output}, "build/generated_copy_2.o");

    return 0;
}
