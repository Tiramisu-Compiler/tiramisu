#include <tiramisu/tiramisu.h>

#include "wrapper_test_205.h"

using namespace tiramisu;

/**
 * Test interchange for non-rectangular loop spaces
 */

void generate_function(std::string name, int size0, int val0)
{
    tiramisu::init(name);

    // -------------------------------------------------------
    // Layer I
    // ------------------------------------------------------- 

    //Iteration variables    
    var i("i"), j("j"), k("k"), l("l"), m("m");
    

    //inputs
    input A("A", {i, i}, p_float64);


    //Computations
    computation A_sub("{A_sub[i,j,k]: 0<=i<40 and 0<=j<i and 0<=k<j}", expr(), true, p_float64, global::get_implicit_function());
    A_sub.set_expression(A_sub(i,j,k) - A(i,k)*A(k,j));
    computation A_div("{A_div[i,j]: 0<=i<40 and 0<=j<i}", expr(), true, p_float64, global::get_implicit_function());
    A_div.set_expression(A_sub(i,j,0)/A_sub(j,j,0));
    computation A_out("{A_out[i,l,m]: 0<=i<40 and i<=l<40 and 0<=m<i}", expr(), true, p_float64, global::get_implicit_function());
    A_out.set_expression(A_out(i,l,m) - A_div(i,m)*A_div(m,l));

    
    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    A_sub.then(A_div,j)
         .then(A_out, i);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    //Input Buffers
    buffer b_A("b_A", {size0,size0}, p_float64, a_output);    

    //Store inputs
    A.store_in(&b_A);    

    //Store computations
    A_sub.store_in(&b_A, {i,j});
    A_div.store_in(&b_A);
    A_out.store_in(&b_A, {i,l});

    // legality check of function
    prepare_schedules_for_legality_checks();
    // analysis
    performe_full_dependency_analysis();     

    // Interchange
    A_out.interchange(0, 1);

    assert(check_legality_of_function() == true);

    // Code generation
    tiramisu::codegen({&b_A}, "build/generated_fct_test_" + std::string(TEST_NUMBER_STR) + ".o");
}

int main(int argc, char **argv)
{
    generate_function("tiramisu_generated_code", SIZE0, 0);

    return 0;
}
