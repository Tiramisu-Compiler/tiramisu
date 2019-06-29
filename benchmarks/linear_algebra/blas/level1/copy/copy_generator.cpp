
#include <tiramisu/tiramisu.h>
#include "benchmarks.h"

/*
  this is an implementation of the BLAS COPY (copies a vector to another vector) function in Tiramisu 
  the performed operation is :
   a = x

   a : size N output vector
   x : size N input vector
   
   The C version of this function is as follow : 
   
    for(int i=0; i<N:i++){
          a(i)=x(i)
    }
   
*/



using namespace tiramisu;

int main(int argc, char **argv)
{
    tiramisu::init("copy");

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------
    constant DIM("N", expr(N));
    input x("x", {"i"}, {DIM}, p_float32);
    var i("i", 0, N);
    computation COPY("copy", {i}, (x(i)));

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    
    COPY.vectorize(i, 32 );    


    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    buffer b_x("b_x", {expr(DIM)}, p_float32, a_input);
    buffer b_a("b_a", {expr(DIM)}, p_float32, a_output);
    x.store_in(&b_x);
    COPY.store_in(&b_a);

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    tiramisu::codegen({&b_a, &b_x}, "generated_copy.o");

    return 0;
}
