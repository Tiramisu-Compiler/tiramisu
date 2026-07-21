#include <tiramisu/tiramisu.h>
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include "function_trisolv_XLARGE_wrapper.h"

using namespace tiramisu;


int main(int argc, char **argv)
{
    tiramisu::init("function_trisolv_XLARGE");

    // -------------------------------------------------------
    // Layer I
    // ------------------------------------------------------- 

    //Iteration variables    
    var i("i", 0, 4000), j("j");
    

    //inputs
    input L("L", {i, i}, p_float64);
    input b("b", {i}, p_float64);
    input x("x", {i}, p_float64);


    //Computations

    computation x_init("{x_init[i]: 0<=i<4000 }", expr(), true, p_float64, global::get_implicit_function());
    x_init.set_expression(b(i));
    computation x_sub("{x_sub[i,j]: 0<=i<4000 and 0<=j<i}", expr(), true, p_float64, global::get_implicit_function());
    x_sub.set_expression(x(i) - L(i,j) * x(j));
    computation x_out("{x_out[i]: 0<=i<4000 }", expr(), true, p_float64, global::get_implicit_function());
    x_out.set_expression(x(i) / L(i,i));

    
    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    x_init.then(x_sub,i)
            .then(x_out, i);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    //Input Buffers
    buffer b_L("b_L", {4000,4000}, p_float64, a_input);    
    buffer b_b("b_b", {4000}, p_float64, a_input);    
    buffer b_x("b_x", {4000}, p_float64, a_output);    

    //Store inputs
    L.store_in(&b_L);
    b.store_in(&b_b);
    x.store_in(&b_x);   

    //Store computations
    x_init.store_in(&b_x);
    x_sub.store_in(&b_x, {i});
    x_out.store_in(&b_x);

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    tiramisu::codegen({&b_L, &b_b, &b_x}, "function_trisolv_XLARGE.o");

    return 0;
}
