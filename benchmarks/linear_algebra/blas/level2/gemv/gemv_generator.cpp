#include <tiramisu/tiramisu.h>

#include "benchmarks.h"

#define UNROLL_FACTOR 64

using namespace tiramisu;

/**
 * Benchmark for BLAS GEMV
 *     out = a*A*x + b*y
 *
 *     A : is a M x N matrix
 *     x : is a size N vector
 *     y : is a size M vector
 *     a,b : are scalars
 *
 *     out : is a size M vector
**
We will make a tiramisu implementation of this code :
  for(int i=0; i<M; i++)
  {
    tmp=0;
    for(int j=0; j<N; j++){
      tmp+= A(i, j) * x(j)
    }
	tmp *= alpha
    result(i)= tmp + beta * y(i)
  }
*/

int main(int argc, char **argv)
{
    tiramisu::init("gemv");

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------
    constant MM("M", expr(M)), NN("N", expr(N));

    //Iteration variables
    var i("i", 0, MM), j("j", 0, NN);

    //Inputs
    input A("A", {i,j}, p_float64);
    input x("x", {j}, p_float64);
    input y("y", {i}, p_float64);
    input alpha("alpha", {}, p_float64);
    input beta("beta", {}, p_float64);

    //Computations
    computation result_init("result_init", {i}, expr(cast(p_float64, 0)), p_float64);
    computation sum_row("sum_row", {i,j}, p_float64);
    sum_row.set_expression(expr(sum_row(i, j-1) + A(i, j) * x(j)));
    computation mult_alpha("mult_alpha", {i}, expr(alpha(0)* sum_row(i, NN-1)), p_float64);
    computation add_y("add_y", {i}, expr(mult_alpha(i) + beta(0) * y(i)), p_float64);

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    add_y.after(mult_alpha, i);
    mult_alpha.after(sum_row, i);
    sum_row.after(result_init, i);
	
    //Unrolling
    sum_row.unroll(j, UNROLL_FACTOR);
    //Parallelization
    sum_row.parallelize(i);
	
    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    //Input Buffers
    buffer buf_A("buf_A", {MM,NN}, p_float64, a_input);
    buffer buf_x("buf_x", {NN}, p_float64, a_input);
    buffer buf_y("buf_y", {MM}, p_float64, a_input);
    buffer buf_alpha("buf_alpha", {1}, p_float64, a_input);
    buffer buf_beta("buf_beta", {1}, p_float64, a_input);

    //Output Buffers
    buffer buf_result("buf_result", {MM}, p_float64, a_output);

    //Store inputs
    A.store_in(&buf_A);
    x.store_in(&buf_x);
    y.store_in(&buf_y);
    alpha.store_in(&buf_alpha);
    beta.store_in(&buf_beta);

    //Store computations
    result_init.store_in(&buf_result);
    sum_row.store_in(&buf_result, {i});
    mult_alpha.store_in(&buf_result);
    add_y.store_in(&buf_result);

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    tiramisu::codegen({&buf_A, &buf_x, &buf_y, &buf_alpha, &buf_beta, &buf_result}, "generated_gemv.o");

    return 0;
}