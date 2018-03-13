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

#include "wrapper_test_86.h"

using namespace tiramisu;

/**
 * Test bound inference. 
 */

void generate_function(std::string name, int size, int val0)
{
    tiramisu::global::set_default_tiramisu_options();
    tiramisu::global::set_loop_iterator_default_data_type(tiramisu::p_int32);

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    tiramisu::function function0(name);
    tiramisu::constant N("N", tiramisu::expr((int32_t) size), p_int32, true, NULL, 0, &function0);
    tiramisu::var x("x"), y("y"), r("r");
    tiramisu::computation input("[N]->{input[y,x]}", tiramisu::expr((uint8_t) val0), true, p_uint8, &function0);
    tiramisu::computation volume("[N]->{volume[r,y,x]: 0<=x<N and 0<=y<N and 0<=r<4 and x>=r}", input(y, x-r), true, p_uint8, &function0);

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    volume.after(input, computation::root);
    function0.compute_bounds();

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    tiramisu::buffer b_input("b_input", {size, size}, tiramisu::p_uint8, a_temporary, &function0);
    input.set_access("[N]->{input[i,j]->b_input[i,j]}");
    tiramisu::buffer b_volume("b_volume", {4, size, size}, tiramisu::p_uint8, a_output, &function0);
    volume.set_access("[N]->{volume[i,j,k]->b_volume[i,j,k]}");

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    function0.set_arguments({&b_volume});
    function0.gen_time_space_domain();
    function0.gen_isl_ast();
    function0.gen_halide_stmt();
    function0.gen_halide_obj("build/generated_fct_test_" + std::string(TEST_NUMBER_STR) + ".o");
}

int main(int argc, char **argv)
{
    generate_function("tiramisu_generated_code", SIZE, 5);

    return 0;
}
