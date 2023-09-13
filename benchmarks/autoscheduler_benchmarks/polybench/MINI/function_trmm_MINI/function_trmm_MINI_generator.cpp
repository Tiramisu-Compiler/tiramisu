#include <tiramisu/tiramisu.h>
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include "function_trmm_MINI_wrapper.h"

using namespace tiramisu;


int main(int argc, char **argv)
{
    tiramisu::init("function_trmm_MINI");

    // -------------------------------------------------------
    // Layer I
    // ------------------------------------------------------- 
    constant NN("NN", 30);
    constant MM("MM", 20);

    //Iteration variables    
//     var i("i", 0, 20), j("j", 0, 30), k("k", 0, 20);
    var i("i", 0, 20), j("j", 0, 30), k("k");
    

    //inputs
    input A("A", {i, k}, p_float64);
    input B("B", {i, j}, p_float64);


    //Computations
    computation AB("{AB[i,j,k]: 0<=i<20 and 0<=j<30 and i+1<=k<20}", expr(), true, p_float64, global::get_implicit_function());
    AB.set_expression(B(i,j) + A(k,i)*B(k,j));
    computation B_out("{B_out[i,j]: 0<=i<20 and 0<=j<30}", expr(), true, p_float64, global::get_implicit_function());
    B_out.set_expression(B(i,j)*1.5);

    
    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    AB.then(B_out, j);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    //Input Buffers
    buffer b_A("b_A", {20,20}, p_float64, a_input);
    buffer b_B("b_B", {20,30}, p_float64, a_output);
    

    //Store inputs
    A.store_in(&b_A);
    B.store_in(&b_B);
    

    //Store computations
    AB.store_in(&b_B, {i,j});
    B_out.store_in(&b_B);

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    tiramisu::codegen({&b_A, &b_B}, "function_trmm_MINI.o");

    return 0;
}
