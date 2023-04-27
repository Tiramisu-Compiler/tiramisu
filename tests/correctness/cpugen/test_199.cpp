#include <tiramisu/tiramisu.h>

#include "wrapper_test_199.h"

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
    tiramisu::var i3("i3"), j3("j3");
    tiramisu::input A("A", {i, j}, p_uint8);

    
    tiramisu::computation result({i,j}, A(i-1, j-1) + A(i-1, j+1)+ A(i-1, j));

    tiramisu::buffer buff_A("buff_A", {N, N}, tiramisu::p_uint8, a_input);
    A.store_in(&buff_A);
    result.store_in(&buff_A);

    // analysis
    perform_full_dependency_analysis();

    // test auto innermost parallism

    function * fct = tiramisu::global::get_implicit_function();

    auto auto_skewing_result = fct->skewing_local_solver_positive({&result},i,j,2);

    auto identity_skewing = std::get<2>(auto_skewing_result);
    //lower triangular skewing to enable tiling

    assert(identity_skewing.size() > 0);

    auto first_sol = identity_skewing[0];

    result.skew(i,j,
        std::get<0>(first_sol),
        std::get<1>(first_sol),
        std::get<2>(first_sol),
        std::get<3>(first_sol), i1,j1);

    result.tile(i1 ,j1 , 4, 4, i2, j2, i3, j3);

    // legality check of function
    prepare_schedules_for_legality_checks();

    assert(check_legality_of_function() == true);

    // Code generation
    tiramisu::codegen({&buff_A}, "generated_fct_test_" + std::string(TEST_NUMBER_STR) + ".o");
}

int main(int argc, char **argv)
{
    generate_function("tiramisu_generated_code", SIZE1, 0);

    return 0;
}
