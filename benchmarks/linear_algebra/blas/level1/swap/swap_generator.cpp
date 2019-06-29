
#include <tiramisu/tiramisu.h>
#include "benchmarks.h"

/*
  this is an implementation of the BLAS SWAP function (swaps a vector with an another vector)  in Tiramisu 
  the performed operation is :
   a = a + b
   b = a - b
   a = a - b

   a : size N vector
   b : size N vector
   
   The C version of this function is as follow : 
   
    for(int i=0; i<N:i++){
          a(i) = a(i) + b(i)
	  b(i) = a(i) - b(i)
	  a(i) = a(i) - b(i)
    }
   
*/



using namespace tiramisu;

int main(int argc, char **argv)
{
    tiramisu::init("swap");

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------
    constant DIM("N", expr(N));
    input a("a", {"i"}, {DIM}, p_float32);
    input b("b", {"i"}, {DIM}, p_float32);
    var i("i", 0, N);
    computation S1("S1", {i}, p_float32);
    S1.set_expression( a(i) + b(i) );
    computation S2("S2", {i}, p_float32);
    S2.set_expression( a(i) - b(i) );
    computation S3("S3", {i}, p_float32);
    S3.set_expression( a(i) - b(i) );

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    
    

    S3.after(S2, i);
    S2.after(S1, i);

    S1.vectorize(i, 32 );  
    S2.vectorize(i, 32 );    
    S3.vectorize(i, 32 ); 

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    buffer b_b("b_b", {expr(DIM)}, p_float32, a_input);
    buffer b_a("b_a", {expr(DIM)}, p_float32, a_input);
    
    b.store_in(&b_b);
    a.store_in(&b_a);

    S1.store_in(&b_a);
    S2.store_in(&b_b);
    S3.store_in(&b_a);

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    tiramisu::codegen({&b_a, &b_b}, "generated_swap.o");

    return 0;
}
