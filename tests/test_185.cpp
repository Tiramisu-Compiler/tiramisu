#include <tiramisu/tiramisu.h>

#include "wrapper_test_185.h"

using namespace tiramisu;

/**
 * Test parallelization legality.
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

    // legality check of function
    prepare_schedules_for_legality_checks();

    assert(loop_parallelization_is_legal(i,{&result}) == false);
    assert(loop_parallelization_is_legal(j,{&result}) == false);

    result.skew(i,j,1,1,i1,j1);

    assert(loop_parallelization_is_legal(j1,{&result}) == true);
    assert(loop_parallelization_is_legal(i1,{&result}) == false);

    result.parallelize(j1);

    assert(check_legality_of_function() == true);

    // Code generation
    tiramisu::codegen({&buff_A}, "build/generated_fct_test_" + std::string(TEST_NUMBER_STR) + ".o");
}

int main(int argc, char **argv)
{
    generate_function("tiramisu_generated_code", SIZE1, 0);

    return 0;
}
