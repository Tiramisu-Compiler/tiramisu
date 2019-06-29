/**
  swap : swap 2 vectors content 


  int m ;
  for (int i = 0; i < 10; i++)
      m = B[i];
      B[i] = A[i];
      A[i]=m;
*/

#include <tiramisu/tiramisu.h>

#define NN 100

using namespace tiramisu;

int main(int argc, char **argv)
{
    tiramisu::init("swap");

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    // Declare the constant N.
    constant N("N", NN);

    // Declare iterator variable.
    var i("i", 0, N);

    // Declare the input.
    input A("A", {i}, p_uint8);
    input B("B", {i}, p_uint8);

    // Declare the output computation.
    computation c1("c1", {i}, A(i) + B(i) );
    computation c2("c2", {i}, A(i) - B(i) );
    computation c3("c3", {i}, A(i) - B(i) );


    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    // Declare input and output buffers.
    buffer b_A("b_A", {expr(NN)}, p_uint8, a_input);
    buffer b_B("b_B", {expr(NN)}, p_uint8, a_input);

    // Map the computations to a buffer.
    A.store_in(&b_A);
    B.store_in(&b_B);
    c1.store_in(&b_A);
    c2.store_in(&b_B);
    c3.store_in(&b_A);
  

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    // Set the arguments and generate code
    tiramisu::codegen({&b_A, &b_B}, "build/generated_fct_developers_swap.o");

    return 0;
}
