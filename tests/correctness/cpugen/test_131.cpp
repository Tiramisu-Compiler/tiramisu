#include <tiramisu/core.h>

#include <string.h>
#include <Halide.h>

#include "wrapper_test_131.h"

using namespace tiramisu;

/**
 * Test skewing command.
 */

void generate_function(std::string name, int size, int val0)
{
    tiramisu::init(name);

    // Layer I
    tiramisu::constant N("N", tiramisu::expr((int32_t) size));

    tiramisu::var i("i", 0, N), j("j", 0, N);
    tiramisu::computation result({i,j}, expr((uint8_t) 2));

    // Scheduling
    result.skew(i, j, 1);

    // Data mapping
    tiramisu::buffer result_buffer("result_buffer", {size, size}, tiramisu::p_uint8, a_output);
    result.store_in(&result_buffer);

    // Code generation
    tiramisu::codegen({&result_buffer}, "build/generated_fct_test_" + std::string(TEST_NUMBER_STR) + ".o");
}

int main(int argc, char **argv)
{
    generate_function("tiramisu_generated_code", SIZE1, 0);

    return 0;
}
