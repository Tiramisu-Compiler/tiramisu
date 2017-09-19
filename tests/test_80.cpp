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

#include "wrapper_test_80.h"

using namespace tiramisu;

/**
 * Test tiling (with dimension names).
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
    tiramisu::var i1("i1");
    tiramisu::var j1("j1");
    tiramisu::var i2("i2");
    tiramisu::var j2("j2");

    tiramisu::computation S0("[N]->{S0[i,j]: 0<=i<N and 0<=j<N}", tiramisu::expr((uint8_t) val0), true, p_uint8, &function0);
    tiramisu::computation S1("[N]->{S1[i,j]: 0<=i<N and 0<=j<N}", S0(i,j), true, p_uint8, &function0);

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    S0.tile(i, j, 4, 4, i1, j1, i2, j2);
    S1.tile(0, 1, 4, 4);

    S1.after(S0, computation::root_dimension);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    function0.allocate_and_map_buffers_automatically();

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    function0.set_arguments({S1.get_automatically_allocated_buffer()});
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
