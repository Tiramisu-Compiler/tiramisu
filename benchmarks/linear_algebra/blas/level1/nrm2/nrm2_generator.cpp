

#include <tiramisu/tiramisu.h>
#include "benchmarks.h"
#include "math.h"
/*
  IMPLEMENTATION OF THE NMR2 FUNCTION IN TIRAMISU 
  
   out = sqrt(x*'x) 

   where : 
   x : is a size N vector and 'x is the transpose of x 
   
*/



using namespace tiramisu;

int main(int argc, char **argv)
{
    tiramisu::init("nrm2");

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    constant NN("N", expr(N));

    var i("i", 0, NN);
    input x("x", {"i"}, {NN},p_float64);
    computation S_init("S_init", {}, expr(cast(p_float64,0)));
    computation S("S", {i},p_float64);
    S.set_expression( S(i-1) + x(i)*x(i));
    computation result("result", {},p_float64);
    result.set_expression(cast(p_float64,expr(o_sqrt,S(NN),p_float64))); 
     
    



    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

     S.after(S_init,{}); //
     result.after(S,computation::root); //

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    buffer b_x("b_x", {expr(NN)}, p_float64, a_input);   
    buffer b_result("b_result", {expr(1)}, p_float64, a_output);

    x.store_in(&b_x);
    S_init.store_in(&b_result,{});
    S.store_in(&b_result,{});
    result.store_in(&b_result,{});

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    tiramisu::codegen({&b_x,&b_result}, "generated_nrm2.o");

    return 0;
}
