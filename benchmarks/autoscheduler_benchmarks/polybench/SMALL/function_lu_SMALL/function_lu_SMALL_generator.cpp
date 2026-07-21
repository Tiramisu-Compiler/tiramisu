#include <tiramisu/tiramisu.h>
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include "function_lu_SMALL_wrapper.h"

using namespace tiramisu;

int main(int argc, char **argv)
{
    tiramisu::init("function_lu_SMALL");

    // -------------------------------------------------------
    // Layer I
    // ------------------------------------------------------- 

    //Iteration variables    
    var i("i"), j("j"), k("k"), l("l"), m("m");
    

    //inputs
    input A("A", {i, i}, p_float64);


    //Computations
    computation A_sub("{A_sub[i,j,k]: 0<=i<120 and 0<=j<i and 0<=k<j}", expr(), true, p_float64, global::get_implicit_function());
    A_sub.set_expression(A(i,j) - A(i,k)*A(k,j));
    computation A_div("{A_div[i,j]: 0<=i<120 and 0<=j<i}", expr(), true, p_float64, global::get_implicit_function());
    A_div.set_expression(A(i,j)/A(j,j));
    computation A_out("{A_out[i,l,m]: 0<=i<120 and i<=l<120 and 0<=m<i}", expr(), true, p_float64, global::get_implicit_function());
    A_out.set_expression(A(i,l) - A(i,m)*A(m,l));

    
    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    A_sub.then(A_div,j)
         .then(A_out, i);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    //Input Buffers
    buffer b_A("b_A", {120,120}, p_float64, a_output);    

    //Store inputs
    A.store_in(&b_A);    

    //Store computations
    A_sub.store_in(&b_A, {i,j});
    A_div.store_in(&b_A);
    A_out.store_in(&b_A, {i,l});


    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    tiramisu::codegen({&b_A}, "function_lu_SMALL.o");

    return 0;
}
