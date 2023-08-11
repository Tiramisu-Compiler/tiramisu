#include <tiramisu/tiramisu.h>
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include "function_syrk_XLARGE_wrapper.h"

using namespace tiramisu;


int main(int argc, char **argv)
{
    tiramisu::init("function_syrk_XLARGE");

    // -------------------------------------------------------
    // Layer I
    // ------------------------------------------------------- 
    constant NN("NN", 2600);
    constant MM("MM", 2000);

    //Iteration variables    
//     var i("i", 0, 2600), j("j", 0, 2600), k("k", 0, 2000);
    var i("i", 0, 2600), j("j"), k("k", 0, 2000);

    //inputs
    input A("A", {i, k}, p_float64);
    input C("C", {i, j}, p_float64);


    //Computations
    computation C_beta("{C_beta[i,j]: 0<=i<2600 and 0<=j<=i}", expr(), true, p_float64, global::get_implicit_function());
    C_beta.set_expression(C(i,j)*1.2);
    computation C_out("{C_out[i,k,j]: 0<=i<2600 and 0<=j<=i and 0<=k<2000}", expr(), true, p_float64, global::get_implicit_function());
    C_out.set_expression(C(i,j)+ A(i,k)*A(j,k)*1.5);

    
    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    C_beta.then(C_out, i);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    //Input Buffers
    buffer b_A("b_A", {2600,2000}, p_float64, a_input);
    buffer b_C("b_C", {2600,2600}, p_float64, a_output);
    

    //Store inputs
    A.store_in(&b_A);
    C.store_in(&b_C);
    

    //Store computations
    C_beta.store_in(&b_C);
    C_out.store_in(&b_C, {i,j});

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    tiramisu::codegen({&b_A, &b_C}, "function_syrk_XLARGE.o");

    return 0;
}
