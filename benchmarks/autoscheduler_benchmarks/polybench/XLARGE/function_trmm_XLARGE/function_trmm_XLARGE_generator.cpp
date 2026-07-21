#include <tiramisu/tiramisu.h>
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include "function_trmm_XLARGE_wrapper.h"

using namespace tiramisu;


int main(int argc, char **argv)
{
    tiramisu::init("function_trmm_XLARGE");

    // -------------------------------------------------------
    // Layer I
    // ------------------------------------------------------- 
    constant NN("NN", 2600);
    constant MM("MM", 2000);

    //Iteration variables    
//     var i("i", 0, 2000), j("j", 0, 2600), k("k", 0, 2000);
    var i("i", 0, 2000), j("j", 0, 2600), k("k");
    

    //inputs
    input A("A", {i, k}, p_float64);
    input B("B", {i, j}, p_float64);


    //Computations
    computation AB("{AB[i,j,k]: 0<=i<2000 and 0<=j<2600 and i+1<=k<2000}", expr(), true, p_float64, global::get_implicit_function());
    AB.set_expression(B(i,j) + A(k,i)*B(k,j));
    computation B_out("{B_out[i,j]: 0<=i<2000 and 0<=j<2600}", expr(), true, p_float64, global::get_implicit_function());
    B_out.set_expression(B(i,j)*1.5);

    
    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    AB.then(B_out, j);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    //Input Buffers
    buffer b_A("b_A", {2000,2000}, p_float64, a_input);
    buffer b_B("b_B", {2000,2600}, p_float64, a_output);
    

    //Store inputs
    A.store_in(&b_A);
    B.store_in(&b_B);
    

    //Store computations
    AB.store_in(&b_B, {i,j});
    B_out.store_in(&b_B);

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    tiramisu::codegen({&b_A, &b_B}, "function_trmm_XLARGE.o");

    return 0;
}
