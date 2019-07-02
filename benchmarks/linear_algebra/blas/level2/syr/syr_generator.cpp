#include <tiramisu/tiramisu.h>
#include "benchmarks.h"

using namespace tiramisu;

/*
    SYR:
    ----
    performs the symmetric rank 1 operation
        A := alpha*x*x**T + A,

    where:
        alpha is a real scalar,
        x is an n element vector,
        A is an n by n symmetric matrix.

    The C version of this function is as follow :
        for(int i=0; i<N:i++){
            for(int j=0; j<N; j++){
	            result[i][j] = alpha*x[i] * x[j] + A[i][j];
            }
        }
*/

int main(int argc, char **argv)
{
    tiramisu::init("syr");

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    // Constant
    constant NN("NN", expr(N));

    // Iterators
    var i("i", 0, NN), j("j", 0, NN);

    // scalar
    input alpha("alpha", {}, p_float64);

    // The N by N input matrix
    input A("A", {i, j}, p_float64);

    // The input vector x
    input x("x", {i}, p_float64);

    // Computations
    computation mul_x_xt("mul_x_xt", {i, j}, p_float64);
    computation sum_all("sum_all", {i, j}, p_float64);

    mul_x_xt.set_expression(alpha(0) * x(i) * x(j));
    sum_all.set_expression(cast(p_float64, mul_x_xt(i, j)) + A(i, j));

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    sum_all.after(mul_x_xt, i);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    // Input buffers
    buffer b_a("b_a", {expr(NN)}, p_float64, a_input);
    buffer b_x("b_x", {expr(NN)}, p_float64, a_input);
    buffer b_alpha("b_alpha", {expr(1)}, p_float64, a_input);

    // Output buffer
    buffer b_result("b_result", {expr(NN), expr(NN)}, p_float64, a_output);

    // Storing input
    A.store_in(&b_a, {i, j});
    x.store_in(&b_x, {i});
    alpha.store_in(&b_alpha);

    // Storing output
    mul_x_xt.store_in(&b_result);
    sum_all.store_in(&b_result, {i, j});

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    tiramisu::codegen({&b_a, &b_x, &b_alpha, &b_result}, "generated_syr.o");

    return 0;
}