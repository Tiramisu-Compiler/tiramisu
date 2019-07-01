/* 
    This program shows how to process BLAS LEVEL2 syr2

    for i = 0 .. N
        for j = 0 .. N
            C[i,j] = A[i,j] + alpha * X[i]*Y[j] + alpha * X[j]*Y[i]
     
*/

#include <tiramisu/tiramisu.h>

#define NN 100
#define alpha 3 

using namespace tiramisu;

int main(int argc, char **argv)
{
    tiramisu::init("syr2");

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------
    constant N("N", NN);
    var i("i", 0, N), j("j", 0, N);

    // Declare inputs : A(Matrix N*N) , X(Vector dim=N) , Y(Vector dim=N)
    input A("A", {i, j}, p_uint8);
    input x("B", {i}, p_uint8);
    input y("C", {j}, p_uint8);

    // Declare output which is result of computation
    computation output("output", {i,j}, expr((uint8_t)alpha) * x(i) * y(j) + expr((uint8_t)alpha) * x(j) * y(i)  + A(i, j) );
    
    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    // Declare the buffers.
    buffer b_A("b_A", {N,N}, p_uint8, a_input);
    buffer b_x("b_x", {N}, p_uint8, a_input);
    buffer b_y("b_y", {N}, p_uint8, a_input);
    buffer b_output("b_output", {N,N}, p_uint8, a_output);

    // Map the computations to a buffer.
    A.store_in(&b_A);
    x.store_in(&b_x);
    y.store_in(&b_y);
    output.store_in(&b_output,{i,j});
 
    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    tiramisu::codegen({&b_A, &b_x, &b_y, &b_output}, "build/generated_fct_developers_syr2.o");

    return 0;
}
