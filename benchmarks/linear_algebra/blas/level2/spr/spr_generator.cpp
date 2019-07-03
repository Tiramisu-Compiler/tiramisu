/* 
    This program shows how to process BLAS LEVEL2 SPR

    for i = 0 .. N
        for j = 0 .. N
            C[i,j] = A[i,j] + alpha * X[i]*X[j] 
     
*/

#include <tiramisu/tiramisu.h>
#include "benchmarks.h"

using namespace tiramisu;

int main(int argc, char **argv)
{
    tiramisu::init("spr");

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------
    constant NN("NN", N);
    var i("i", 0, NN), j("j", 0, NN);

    // Declare inputs : A(Matrix N*N) , X(Vector dim=N), alpha(scalar)
    input A("A", {i, j}, p_float64);
    input x("B", {i}, p_float64);
    input alpha("alpha", {}, p_float64);

    // Declare output which is result of computation
    computation output("output", {i,j}, cast(p_float64, alpha(0) * x(j) * x(i)  + A(i, j) ));
   
    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    // Declare the buffers.
    buffer b_A("b_A", {expr(NN),expr(NN)}, p_float64, a_input);
    buffer b_x("b_x", {expr(NN)}, p_float64, a_input);
    buffer b_alpha("b_alpha", {expr(1)}, p_float64, a_input);
    buffer b_output("b_output", {expr(NN),expr(NN)}, p_float64, a_output);

    // Map the computations to a buffer.
    A.store_in(&b_A);
    x.store_in(&b_x);
    alpha.store_in(&b_alpha);
    output.store_in(&b_output,{i,j});
 
    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    tiramisu::codegen({&b_A, &b_x, &b_alpha, &b_output}, "generated_spr.o");

    return 0;
}
