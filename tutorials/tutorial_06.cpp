#include <isl/set.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>

#include <tiramisu/debug.h>
#include <tiramisu/core.h>

#include <string.h>
#include <Halide.h>

#include "wrapper_tutorial_06.h"

using namespace tiramisu;

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
 * {result[0]       }: 0
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

void generate_function(std::string name, int size, int val0)
{
    tiramisu::global::set_default_tiramisu_options();

    tiramisu::function function0(name);
    tiramisu::constant N("N", tiramisu::expr((int32_t) size), p_int32, true, NULL, 0, &function0);

    tiramisu::buffer input_buffer("input_buffer", 1, {size}, tiramisu::p_uint8, NULL, a_input, &function0);
    tiramisu::buffer result_scalar("result_scalar", 1, {1}, tiramisu::p_uint8, NULL, a_output, &function0);

    tiramisu::var i = tiramisu::var("i");
    tiramisu::computation input("[N]->{input[i]}", tiramisu::expr(), false, p_uint8, &function0);
    tiramisu::computation result("[N]->{result[0]}", tiramisu::expr(input(0)), true, p_uint8, &function0);
    tiramisu::computation *result1 = result.add_update("[N]->{result[i]: 1<=i<N}", (result(i-1) + input(i)), true, p_uint8, &function0);

    input.set_access("[N]->{input[i]->input_buffer[i]}");
    result.set_access("[N]->{result[i]->result_scalar[0]}");
    result1->set_access("[N]->{result[i]->result_scalar[0]}");

    result1->after(result, computation::root_dimension);

    function0.set_arguments({&input_buffer, &result_scalar});
    function0.gen_time_processor_domain();
    function0.gen_isl_ast();
    function0.gen_halide_stmt();
    function0.gen_c_code();
    function0.gen_halide_obj("build/generated_fct_tutorial_06.o");
}

int main(int argc, char **argv)
{
    generate_function("tiramisu_generated_code", SIZE1, 0);

    return 0;
}
