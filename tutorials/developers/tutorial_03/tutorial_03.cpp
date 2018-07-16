/* Sequence of computations.

for (i = 0; i < M; i++)
  S0(i) = 4;
  S1(i) = 3;
  for (j = 0; j < N; j++)
    S2(i, j) = 2;
  S3(i) = 1;
*/

#include <tiramisu/tiramisu.h>
#define SIZE0 10

using namespace tiramisu;

int main(int argc, char **argv)
{
    // Set default tiramisu options.
    tiramisu::init();

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    function sequence("sequence");
    expr e_M = expr((int32_t) SIZE0);
    constant M("M", e_M, p_int32, true, NULL, 0, &sequence);
    computation c0("[M]->{c0[i]: 0<=i<M}", expr((uint8_t) 4), true, p_uint8, &sequence);
    computation c1("[M]->{c1[i]: 0<=i<M}", expr((uint8_t) 3), true, p_uint8, &sequence);
    computation c2("[M]->{c2[i,j]: 0<=i<M and 0<=j<M}", expr((uint8_t) 2), true, p_uint8,
                   &sequence);
    computation c3("[M]->{c3[i]: 0<=i<M}", expr((uint8_t) 1), true, p_uint8, &sequence);

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    var i("i");
    c1.after(c0, i);
    c2.after(c1, i);
    c3.after(c2, i);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    buffer b0("b0", {expr(SIZE0)}, p_uint8, a_output, &sequence);
    buffer b1("b1", {expr(SIZE0)}, p_uint8, a_output, &sequence);
    buffer b2("b2", {expr(SIZE0), expr(SIZE0)}, p_uint8, a_output, &sequence);
    buffer b3("b3", {expr(SIZE0)}, p_uint8, a_output, &sequence);

    c0.store_in(&b0);
    c1.store_in(&b1);
    c2.store_in(&b2);
    c3.store_in(&b3);

    // -------------------------------------------------------
    // Code Generator
    // -------------------------------------------------------

    sequence.codegen({&b0, &b1, &b2, &b3}, "build/generated_fct_developers_tutorial_03.o");

    return 0;
}
