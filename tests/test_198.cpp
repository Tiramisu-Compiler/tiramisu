#include <tiramisu/tiramisu.h>

#include "wrapper_test_198.h"

using namespace tiramisu;

/**
 * Test skewing command.
 */

void generate_function(std::string name, int size, int val0)
{
    tiramisu::init(name);

    // Algorithm
    tiramisu::constant N("N", tiramisu::expr((int32_t) size));
    tiramisu::var i("i", 1, N-1), j("j", 1, N-2);
    tiramisu::input A("A", {i, j}, p_uint8);

    
    tiramisu::computation result({i,j}, A(i-1, j-1) + A(i-1, j+1));

    // Schedule
    tiramisu::var ni("ni"), nj("nj"), nj2("nj2");

    tiramisu::buffer buff_A("buff_A", {N, N}, tiramisu::p_uint8, a_input);
    A.store_in(&buff_A);
    result.store_in(&buff_A);

    prepare_schedules_for_legality_checks();
    performe_full_dependency_analysis();

    tiramisu::function * f = global::get_implicit_function();
    std::vector<int> res = f->polyhedral_full_solver_positive({&result}, 0, 1, false);
    
    result.skew(0, 1, res[0], res[1], res[2], res[3]);

    result.tile(0, 1, 4, 4);
    // skewing enabled tiling
    assert(check_legality_of_function() == true);
    tiramisu::codegen({&buff_A}, "build/generated_fct_test_" + std::string(TEST_NUMBER_STR) + ".o");
}

int main(int argc, char **argv)
{
    generate_function("tiramisu_generated_code", SIZE1, 0);
    return 0;
}
