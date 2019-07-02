#include <tiramisu/tiramisu.h>
#include "benchmarks.h"

using namespace tiramisu;

/**
*   Benchmark for BLAS SGER :   
*  
*   A = A + a * x * y' 
*
*   where : 
*   A   : a N by M matrix
*   x   : size N vector 
*   y   : size M vector
*   y'  : transpose of Y     
*   a   : scalar 
*   
*   The C code of this function is as follow : 
*   
*   for(int i = 0; i < N; i++){
*       for(int j = 0; j < M; j++){
*	        A[i][j] += a * x[i] * y[j]; 
*        }
*    }
*/

int main(int argc, char **argv)
{
    tiramisu::init("sger");

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    constant NN("N", expr(N));
    constant MM("M", expr(M));

    var i("i", 0, N), j("j", 0, M);

    input A("A", {"i", "j"}, {NN, MM}, p_float64);
    input x("x", {"i"}, {NN}, p_float64);
    input y("y", {"j"}, {MM}, p_float64);
    input alpha("alpha", {}, p_float64);
    computation C("C", {i,j}, p_float64);
    C.set_expression(A(i, j) + alpha(0) * x(i) * y(j));

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    C.parallelize(i);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    buffer b_A("b_A", {expr(NN), expr(MM)}, p_float64, a_output);
    buffer b_x("b_x", {expr(NN)}, p_float64, a_input);   
    buffer b_y("b_y", {expr(MM)}, p_float64, a_input);
    buffer b_alpha("b_alpha", {expr(1)}, p_float64, a_input);

    A.store_in(&b_A);
    x.store_in(&b_x);
    y.store_in(&b_y);
    alpha.store_in(&b_alpha);

    C.store_in(&b_A, {i,j});

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    tiramisu::codegen({&b_A, &b_x, &b_y, &b_alpha}, "generated_sger.o");

    return 0;
}
