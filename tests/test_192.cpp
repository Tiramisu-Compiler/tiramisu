#include <tiramisu/tiramisu.h>

#include "wrapper_test_192.h"

using namespace tiramisu;

/**
 * Test auto skewing command.
 */

void generate_function(std::string name, int size, int val0)
{
    tiramisu::init(name);

    // Algorithm
    tiramisu::constant N("N", tiramisu::expr((int32_t) size));
    tiramisu::var i("i", 1, N-1), j("j", 1, N-1);
    tiramisu::var i1("i1"), j1("j1");
    tiramisu::var i2("i2"), j2("j2");
    tiramisu::input A("A", {i, j}, p_uint8);

    
    tiramisu::computation result({i,j}, A(i-1, j) + A(i, j-1));

    tiramisu::buffer buff_A("buff_A", {N, N}, tiramisu::p_uint8, a_input);
    A.store_in(&buff_A);
    result.store_in(&buff_A);

    // analysis
    perform_full_dependency_analysis();     

    // test auto innermost parallism

    function * fct = tiramisu::global::get_implicit_function();

    auto auto_skewing_result = fct->skewing_local_solver({&result},i,j,2);

    auto inner_parallelism = std::get<1>(auto_skewing_result);
    //second vector of inner parallism

    auto outer_para = std::get<0>(auto_skewing_result);
    //first vector of outer parallism

    assert(outer_para.size() == 0);//no outer_parallelism

    assert(inner_parallelism.size() > 0);

    auto first_sol = inner_parallelism[0];

    result.skew(i,j,first_sol.first,first_sol.second,i1,j1);

    assert(fct->loop_parallelization_is_legal(j1,{&result}) == true);

    // legality check of function
    prepare_schedules_for_legality_checks() ;

    assert(check_legality_of_function() == true) ;

    // Code generation
    tiramisu::codegen({&buff_A}, "build/generated_fct_test_" + std::string(TEST_NUMBER_STR) + ".o");
}

int main(int argc, char **argv)
{
    generate_function("tiramisu_generated_code", SIZE1, 0);

    return 0;
}
