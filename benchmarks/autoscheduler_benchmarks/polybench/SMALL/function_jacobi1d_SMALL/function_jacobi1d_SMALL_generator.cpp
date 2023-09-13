#include <tiramisu/tiramisu.h>
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include "function_jacobi1d_SMALL_wrapper.h"

using namespace tiramisu;


int main(int argc, char **argv)
{
    tiramisu::init("function_jacobi1d_SMALL");

    // -------------------------------------------------------
    // Layer I
    // ------------------------------------------------------- 

    //Iteration variables    
    var i_f("i_f", 0, 120);
    var t("t", 0, 40), i("i", 1, 120-1);
    
    //inputs
    input A("A", {i_f}, p_float64);
    input B("B", {i_f}, p_float64);


    //Computations
    computation comp_B("comp_B", {t,i}, (A(i-1) + A(i) + A(i + 1))*0.33333);
    computation comp_A("comp_A", {t,i}, (B(i-1) + B(i) + B(i + 1))*0.33333);

    // -------------------------------------------------------
    // Layer II
    // ----------  ---------------------------------------------
    comp_B.then(comp_A,t);
    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    //Input Buffers
    buffer b_A("b_A", {120}, p_float64, a_output);    
    buffer b_B("b_B", {120}, p_float64, a_output);

    //Store inputs
    A.store_in(&b_A);
    B.store_in(&b_B);

    //Store computations
    comp_B.store_in(&b_B, {i});
    comp_A.store_in(&b_A, {i});
    

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    tiramisu::codegen({&b_A,&b_B}, "function_jacobi1d_SMALL.o");

    return 0;
}