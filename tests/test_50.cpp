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

#include "wrapper_test_50.h"
#include "../include/tiramisu/core.h"
#include "../include/tiramisu/expr.h"

using namespace tiramisu;

/**
 * Test allocate_and_map_buffers_automatically().
 */

void generate_function(std::string name, int size, int val0)
{
    tiramisu::global::set_default_tiramisu_options();

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    tiramisu::function function0(name);
    tiramisu::constant N("N", tiramisu::expr((int32_t) size), p_int32, true, NULL, 0, &function0);
    tiramisu::var i("i");
    tiramisu::var j("j");
    tiramisu::computation S0("[N]->{S0[i,j]: 0<=i<N and 0<=j<N}", tiramisu::expr((uint8_t) val0), true, p_uint8, &function0);
    tiramisu::computation S1("[N]->{S1[i,j]: 0<=i<N and 0<=j<N}", S0(i,j), true, p_uint8, &function0);
    tiramisu::computation S2("[N]->{S2[i,j]: 0<=i<N and 0<=j<N}", S1(i,j), true, p_uint8, &function0);
    tiramisu::computation S3("[N]->{S3[i,j]: 0<=i<N and 0<=j<N}", S2(i,j), true, p_uint8, &function0);

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    S1.after(S0,computation::root);
    S2.after(S1,computation::root);
    S3.after(S2,computation::root);

    S0.tag_parallel_level(i);
    S1.tag_parallel_level(j);
    S2.tag_parallel_level(j);
    S3.tag_parallel_level(i);
    S3.tag_parallel_level(j);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    function0.allocate_and_map_buffers_automatically();

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    function0.set_arguments({S3.get_automatically_allocated_buffer()});
    function0.gen_time_space_domain();
    function0.gen_isl_ast();
    function0.gen_halide_stmt();
    function0.gen_halide_obj("build/generated_fct_test_" + std::string(TEST_NUMBER_STR) + ".o");
}

int main(int argc, char **argv)
{
    generate_function("tiramisu_generated_code", SIZE1, 5);

    return 0;
}
