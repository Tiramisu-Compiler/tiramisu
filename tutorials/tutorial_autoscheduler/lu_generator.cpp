#include <tiramisu/tiramisu.h>
#include "polybench-tiramisu.h"
#include "lu.h"

using namespace tiramisu;

/*
LU decomposition without pivoting.
It takes the following as input,
    • A: NxN matrix
and gives the following as outputs:
    • L: NxN lower triangular matrix
    • U: NxN upper triangular matrix
such that A = LU.
*/

int main(int argc, char **argv)
{
    tiramisu::init("lu");

    // -------------------------------------------------------
    // Layer I
    // ------------------------------------------------------- 
    constant NN("NN", N);

    //Iteration variables    
    var i("i"), j("j"), k("k"), l("l"), m("m");
    

    //inputs
    input A("A", {i, i}, p_float64);


    //Computations
    computation A_sub("{A_sub[i,j,k]: 0<=i<128 and 0<=j<i and 0<=k<j}", expr(A_sub(i,j,k) - A(i,k)*A(k,j)), true, p_float64, global::get_implicit_function());
    //A_sub.set_expression(A_sub(i,j,k) - A(i,k)*A(k,j));
    computation A_div("{A_div[i,j]: 0<=i<128 and 0<=j<i}", expr(A_sub(i,j,0)/A_sub(j,j,0)), true, p_float64, global::get_implicit_function());
    //A_div.set_expression(A_sub(i,j,0)/A_sub(j,j,0));
    computation A_out("{A_out[i,l,m]: 0<=i<128 and i<=l<128 and 0<=m<i}", expr(A_out(i,l,m) - A_div(i,m)*A_div(m,l)), true, p_float64, global::get_implicit_function());
    //A_out.set_expression(A_out(i,l,m) - A_div(i,m)*A_div(m,l));

    
    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    A_sub.then(A_div,j)
         .then(A_out, i);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    //Input Buffers
    buffer b_A("b_A", {128,128}, p_float64, a_output);    

    //Store inputs
    A.store_in(&b_A);    

    //Store computations
    A_sub.store_in(&b_A, {i,j});
    A_div.store_in(&b_A);
    A_out.store_in(&b_A, {i,l});

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    tiramisu::codegen({&b_A}, "generated_lu.o");

    return 0;
}
