#include <tiramisu/tiramisu.h>

#include "wrapper_test_194.h"

using namespace tiramisu;

/**
 * Test vectorization legality
 */

void generate_function(std::string name, int size, int val0)
{
    tiramisu::init(name);

    // Algorithm
    tiramisu::var i("i", 1, size-2), j("j", 1, size-2);
    tiramisu::var i1("i1"), j1("j1");
    tiramisu::var i2("i2"), j2("j2");
    tiramisu::input A("A", {i, j}, p_uint8);

    
    tiramisu::computation result({i,j}, A(i-1, j-1) + A(i+1, j+1));

    tiramisu::buffer buff_A("buff_A", {size, size}, tiramisu::p_uint8, a_input);
    A.store_in(&buff_A);
    result.store_in(&buff_A);

    // legality check of function
    prepare_schedules_for_legality_checks();
    // analysis
    perform_full_dependency_analysis();     


    function * fct = tiramisu::global::get_implicit_function();

    auto auto_skewing_result = fct->skewing_local_solver({&result},i,j,2);

    auto outer_parallelism = std::get<0>(auto_skewing_result);
    //first vector of outer parallism

    assert(outer_parallelism.size() > 0);

    auto first_sol = outer_parallelism[0];

    result.skew(i,j,first_sol.first,first_sol.second,i1,j1);

    assert(fct->loop_parallelization_is_legal(i1,{&result}) == true);

    assert(check_legality_of_function() == true);

    result.parallelize(i1);

    // Code generation
    tiramisu::codegen({&buff_A}, "build/generated_fct_test_" + std::string(TEST_NUMBER_STR) + ".o");
}

int main(int argc, char **argv)
{
    generate_function("tiramisu_generated_code", SIZE1, 0);

    return 0;
}
