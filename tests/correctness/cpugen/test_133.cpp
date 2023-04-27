#include <tiramisu/tiramisu.h>

#include "wrapper_test_133.h"

using namespace tiramisu;

/**
 * Test skewing command.
 */

void generate_function(std::string name, int size, int val0)
{
    tiramisu::init(name);

    // Algorithm
    tiramisu::constant N("N", tiramisu::expr((int32_t) size));
    tiramisu::var i("i", 1, N-1), j("j", 1, N-1);
    tiramisu::input A("A", {i, j}, p_uint8);

    tiramisu::computation result({i,j}, A(i-1, j) + A(i, j-1));

    // Schedule
    tiramisu::var ni("ni"), nj("nj");
    result.skew(i, j, 1, ni, nj);
    result.parallelize(ni);

    tiramisu::buffer buff_A("buff_A", {N, N}, tiramisu::p_uint8, a_input);
    A.store_in(&buff_A);
    result.store_in(&buff_A);

    // Code generation
    tiramisu::codegen({&buff_A}, "build/generated_fct_test_" + std::string(TEST_NUMBER_STR) + ".o");
}

int main(int argc, char **argv)
{
    generate_function("tiramisu_generated_code", SIZE1, 0);

    return 0;
}
