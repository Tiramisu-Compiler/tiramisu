#include <tiramisu/tiramisu.h>

#include "benchmarks.h"

#include "math.h"

using namespace tiramisu;

/***
  Benchmark for the BLAS NRM2 
 
  out = sqrt(x*'x) 
  where : 
  x : is a size N vector
  x': is the transpose of x 
***/

int main(int argc, char **argv)
{
    tiramisu::init("nrm2");

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------
  
    constant NN("N", expr(N));
    
    //Iteration variable
    var i("i", 0, NN);
  
    //Input
    input x("x", {"i"}, {NN}, p_float64);
  
    //Computations
    computation S_init("S_init", {}, expr(cast(p_float64, 0)));
    computation S("S", {i}, p_float64);
    computation result("result", {}, p_float64);
    S.set_expression(S(i-1) + x(i ) * x(i));
    result.set_expression(cast(p_float64, expr(o_sqrt, S(NN), p_float64))); 
  
    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
  
     S.after(S_init, {}); 
     result.after(S, computation::root); 
  
    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
  
    //Input Buffer
    buffer b_x("b_x", {expr(NN)}, p_float64, a_input); 
  
    //Output Buffer
    buffer b_result("b_result", {expr(1)}, p_float64, a_output);
    
    //Store input
    x.store_in(&b_x);
  
    //Store computations
    S_init.store_in(&b_result, {});
    S.store_in(&b_result, {});
    result.store_in(&b_result, {});

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    tiramisu::codegen({&b_x, &b_result}, "generated_nrm2.o");

    return 0;
}
