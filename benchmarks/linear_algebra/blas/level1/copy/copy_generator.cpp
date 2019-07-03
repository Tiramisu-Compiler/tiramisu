#include <tiramisu/tiramisu.h>
#include "benchmarks.h"

using namespace tiramisu;

/*
  This is an implementation of the BLAS COPY (copies a vector to another vector) function in Tiramisu 
  The performed operation is :
   a = x
   Where:
   a : size N output vector
   x : size N input vector
   
   The C version of this function is as follow :    
    for(int i=0; i<N:i++)
          a(i)=x(i)   
*/

int main(int argc, char **argv)
{
    tiramisu::init("copy");

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------
    constant DIM("N", expr(N));
    
    //Iteration variables
    var i("i", 0, N);

    //Inputs
    input x("x", {"i"}, {DIM}, p_float32);

    //Computations
    computation COPY("copy", {i}, (x(i)));

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------   
    COPY.vectorize(i, 32);    

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    //Input Buffers
    buffer b_x("b_x", {expr(DIM)}, p_float32, a_input);

    //Output Buffers
    buffer b_a("b_a", {expr(DIM)}, p_float32, a_output);

    //Store inputs
    x.store_in(&b_x);

    //Store computations
    COPY.store_in(&b_a);

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    tiramisu::codegen({&b_a, &b_x}, "generated_copy.o");

    return 0;
}
