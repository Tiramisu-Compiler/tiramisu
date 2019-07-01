#include <tiramisu/tiramisu.h>
#include "benchmarks.h"
#include "math.h"

using namespace tiramisu;

/*
    ASUM:
    ----
    ASUM takes the sum of the absolute values
        result = sum(abs(X))

    Where:
        X is an (1 + (N - 1) * abs(incx)) vector,
        incx storage spacing between elements of X.

    The C version of this function is as follow:
        result = 0;
        for(int i=0; i < N ;i++){
	            result += x[i * incx];
        }
*/

int main(int argc, char **argv)
{
    tiramisu::init("asum");

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------
    constant NN("NN", expr(N));

    // Iterator
    var i("i", 0, NN);

    // inputs
    input incx("incx", {}, p_float64);
    input x("x", {i}, p_float64);

    // Computations
    computation init("init", {}, cast(p_float64, 0));
    computation sum("sum", {i}, p_float64);

    sum.set_expression(sum(i - 1) + expr(o_abs, x(cast(p_float64, i) * cast(p_float64, incx(0)))));

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    // Inputs
    buffer b_incx("b_incx", {expr(1)}, p_float64, a_input);
    buffer b_x("b_x", {expr(NN)}, p_float64, a_input);

    // Output
    buffer b_result("b_result", {expr(1)}, p_float64, a_output);

    incx.store_in(&b_incx);
    x.store_in(&b_x);
    init.store_in(&b_result, {});
    sum.store_in(&b_result, {});

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    tiramisu::codegen({&b_x, &b_incx, &b_result}, "generated_asum.o");

    return 0;
}