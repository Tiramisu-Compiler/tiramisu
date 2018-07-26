/*

This example defines the following sequence of computations.

for (i = 0; i < M; i++)
  S0(i) = 4;
  S1(i) = 3;
  for (j = 0; j < N; j++)
    S2(i, j) = 2;
  S3(i) = 1;
 
 The goal of this tutorial is to show how one can indicate
 the order of computations in Tiramisu.
*/

#include <tiramisu/tiramisu.h>

#define SIZE0 10

using namespace tiramisu;

int main(int argc, char **argv)
{
    tiramisu::init("sequence");

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    constant M("M", expr((int32_t) SIZE0));
  
    var i("i", 0, M),  j("j", 0, M);
  
    // Declare the four computations: c0, c1, c2 and c3.
    computation c0("c0", {i}, expr((uint8_t) 4));
    computation c1("c1", {i}, expr((uint8_t) 3));
    computation c2("c2", {i,j}, expr((uint8_t) 2));
    computation c3("c3", {i}, expr((uint8_t) 1));

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
  
    // By default computations are unordered in Tiramisu. The user has to specify
    // the order exlplicitely (automatic ordering is being developed and will be
    // available soon).
    //
    // The following calls define the order between the computations c3, c2, c1 and c0.
    // c1 is set to be after c0 in the loop level i.  That is, both have the same outer loops
    // up to the loop level i (they share i also) but starting from i, all the
    // computations c1 are ordered after the computations c0.
    c1.after(c0, i);
    c2.after(c1, i);
    c3.after(c2, i);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    buffer b0("b0", {expr(SIZE0)}, p_uint8, a_output);
    buffer b1("b1", {expr(SIZE0)}, p_uint8, a_output);
    buffer b2("b2", {expr(SIZE0), expr(SIZE0)}, p_uint8, a_output);
    buffer b3("b3", {expr(SIZE0)}, p_uint8, a_output);

    c0.store_in(&b0);
    c1.store_in(&b1);
    c2.store_in(&b2);
    c3.store_in(&b3);

    // -------------------------------------------------------
    // Code Generator
    // -------------------------------------------------------

    tiramisu::codegen({&b0, &b1, &b2, &b3}, "build/generated_fct_developers_tutorial_03.o");

    return 0;
}
