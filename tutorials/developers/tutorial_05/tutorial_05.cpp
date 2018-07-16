/**
 * Test reductions.
 *
 * We want to represent the following program
 *
 * result = 0
 * for (int i = 0; i < N; i++)
 *      result = result + input[i];
 *
 * This program computes the sum of all the elements in the
 * buffer input[].
 *
 * In order to implement this program, we create the following two
 * computations
 * {result[0]       }: input(0)
 * {result[i]: 0<i<N}: result(i-1) + input(i)
 *
 * The final result will be in result(N-1).
 *
 * The data mapping should be as follows
 *
 * {result[i]->result_scalar[0]}
 * {input[i]->input_buffer[i]}
 *
 * This means that each computation result[i] is stored
 * in the scalar "result_scalar[0]" and that each computation
 * input[i] is retrieved from input_buffer[i].
 *
 */

#include <tiramisu/tiramisu.h>
#include "wrapper_tutorial_05.h"

using namespace tiramisu;

void generate_function(std::string name, int size, int val0)
{
    // Set default tiramisu options.
    tiramisu::init();

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    function function0(name);
    constant N("N", expr((int32_t) size), p_int32, true, NULL, 0, &function0);
    var i = var("i");
    computation input("[N]->{input[i]}", expr(), false, p_uint8, &function0);
    computation result_init("[N]->{result_init[0]}", expr(input(0)), true, p_uint8, &function0);
    computation result("[N]->{result[i]: 1<=i<N}", expr(), true, p_uint8, &function0);
    result.set_expression((result(i - 1) + input(i)));

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    result.after(result_init, computation::root);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    buffer input_buffer("input_buffer", {size}, p_uint8, a_input, &function0);
    buffer result_scalar("result_scalar", {1}, p_uint8, a_output, &function0);
    input.set_access("[N]->{input[i]->input_buffer[i]}");
    result_init.set_access("[N]->{result_init[i]->result_scalar[0]}");
    result.set_access("[N]->{result[i]->result_scalar[0]}");

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    function0.codegen({&input_buffer, &result_scalar}, "build/generated_fct_developers_tutorial_05.o");
}

int main(int argc, char **argv)
{
    generate_function("tiramisu_generated_code", SIZE1, 0);

    return 0;
}
