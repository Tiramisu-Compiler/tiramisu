#include <tiramisu/tiramisu.h>

#include "wrapper_test_181.h"

using namespace tiramisu;

/**
 * Test skewing command.
 */

void generate_function(std::string name, int size, int val0)
{
    tiramisu::init(name);

    // Algorithm
    tiramisu::constant N("N", tiramisu::expr((int32_t) size));
    tiramisu::var i("i", 1, N-2), j("j", 1, N-2);
    tiramisu::input A("A", {i, j}, p_uint8);

    
    tiramisu::computation result({i,j}, A(i-1, j-1) + A(i+1, j+1));

    tiramisu::computation result2({i,j}, result(i,j) + 1);

    // Schedule
    tiramisu::var ni("ni"), nj("nj"), nj2("nj2");

    tiramisu::buffer buff_A("buff_A", {N, N}, tiramisu::p_uint8, a_input);
    A.store_in(&buff_A);
    result.store_in(&buff_A);
    result2.store_in(&buff_A);

    // analysis
    perform_full_dependency_analysis() ;

    // start adding optimizations
    result.skew(i, j, 1, -1, ni, nj);

    // legality check of function
    prepare_schedules_for_legality_checks() ;

    assert(result.involved_subset_of_dependencies_is_legal(&result2) == true);
    assert(check_legality_of_function() == true) ;

    // Code generation
    tiramisu::codegen({&buff_A}, "generated_fct_test_" + std::string(TEST_NUMBER_STR) + ".o");
}

int main(int argc, char **argv)
{
    generate_function("tiramisu_generated_code", SIZE1, 0);

    return 0;
}
