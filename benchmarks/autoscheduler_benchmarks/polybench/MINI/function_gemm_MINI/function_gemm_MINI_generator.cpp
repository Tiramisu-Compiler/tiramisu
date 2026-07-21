#include <tiramisu/tiramisu.h>
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include "function_gemm_MINI_wrapper.h"


using namespace tiramisu;

int main(int argc, char **argv)
{
    tiramisu::init("function_gemm_MINI");

    //Iteration variables    
    var i("i", 0, 20), j("j", 0, 25), k("k", 0, 30);
    

    //inputs
    input A("A", {i, k}, p_float64);
    input B("B", {k, j}, p_float64);
    input C("C", {i, j}, p_float64);


    //Computations
    
    computation C_init("C_init", {i,j}, C(i,j)*1.2);
    computation C_out("C_out", {i,k,j}, p_float64);
    C_out.set_expression(C(i,j)+A(i,k)*B(k,j)*1.5);
    
    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    C_init.then(C_out, i);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    //Input Buffers
    buffer b_A("b_A", {20,30}, p_float64, a_input);
    buffer b_B("b_B", {30,25}, p_float64, a_input);
    buffer b_C("b_C", {20,25}, p_float64, a_output);
     
    

    //Store inputs
    A.store_in(&b_A);
    B.store_in(&b_B);
    C.store_in(&b_C);
    

    //Store computations
    C_init.store_in(&b_C);
    C_out.store_in(&b_C, {i,j});

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    tiramisu::codegen({&b_A, &b_B, &b_C}, "function_gemm_MINI.o");

    return 0;
}

