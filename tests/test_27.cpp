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

#include "wrapper_test_27.h"

using namespace tiramisu;

/**
 * Test buffer allocation and scheduling.
 */

void generate_function(std::string name, int size, int val0)
{
    tiramisu::global::set_default_tiramisu_options();

    // Layer I
    tiramisu::function function0(name);
    tiramisu::constant N("N", tiramisu::expr((int32_t) size), p_int32, true, NULL, 0, &function0);
    tiramisu::var i = tiramisu::var("i");
    tiramisu::var j = tiramisu::var("j");
    tiramisu::computation temp("[N]->{temp[i,j]: 0<=i<N and 0<=j<N}", (tiramisu::expr((uint8_t) 1)),
                               true, p_uint8, &function0);
    tiramisu::computation result("[N]->{result[i,j]: 0<=i<N and 0<=j<N}", (temp(i, j) + (uint8_t) 1),
                                 true, p_uint8, &function0);
    function0.compute_bounds();

    // Data mapping
    tiramisu::buffer temp_buffer("temp_buffer", 1, {size}, tiramisu::p_uint8, NULL, a_temporary,
                                 &function0);
    tiramisu::buffer result_buffer("result_buffer", 2, {size, size}, tiramisu::p_uint8, NULL, a_output,
                                   &function0);
    temp.set_access("[N]->{temp[i,j]->temp_buffer[j]}");
    result.set_access("[N]->{result[i,j]->result_buffer[i,j]}");
    tiramisu::computation *allocation = temp_buffer.allocate_at(&temp, 0);

    // Scheduling
    allocation->before(temp, 0);
    temp.before(result, 1);

    // Code generation
    function0.set_arguments({&result_buffer});
    function0.gen_time_space_domain();
    function0.gen_isl_ast();
    function0.gen_halide_stmt();
    function0.gen_c_code();
    function0.gen_halide_obj("build/generated_fct_test_" + std::string(TEST_NUMBER_STR) + ".o");
}

int main(int argc, char **argv)
{
    generate_function("tiramisu_generated_code", SIZE1, 0);

    return 0;
}
