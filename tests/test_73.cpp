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

#include "wrapper_test_73.h"

using namespace tiramisu;

/**
 * A test for low level separation.
 */

void generate_function(std::string name)
{
    tiramisu::global::set_default_tiramisu_options();
    tiramisu::global::set_loop_iterator_default_data_type(tiramisu::p_int32);

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    tiramisu::function function0(name);
    tiramisu::constant N("N", tiramisu::expr((int32_t) SIZE), p_int32, true, NULL, 0, &function0);
    tiramisu::var i("i");
    tiramisu::computation S0("[N]->{S0[i]: 0<=i<N}", tiramisu::expr((uint8_t) 5), true, p_uint8, &function0);

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    S0.set_low_level_schedule("[N]->{S0[i]->S0[0, 0, i, 0]: 0<=i<4*floor(N/4); S0[i]->S0[0, 10, i, 0]: 4*floor(N/4)<=i<N}");

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    function0.allocate_and_map_buffers_automatically();

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    function0.set_arguments({S0.get_automatically_allocated_buffer()});
    function0.gen_time_space_domain();
    function0.dump_time_processor_domain();
    function0.gen_isl_ast();
    function0.gen_halide_stmt();
    function0.gen_halide_obj("build/generated_fct_test_" + std::string(TEST_NUMBER_STR) + ".o");
}

int main(int argc, char **argv)
{
    generate_function("tiramisu_generated_code");

    return 0;
}
