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

#include "wrapper_test_96.h"

using namespace tiramisu;

/**
 * This test checks that it is possible to run 64 bit iterator variables.
 */

void generate_function(std::string name)
{
    tiramisu::global::set_default_tiramisu_options();
    global::set_loop_iterator_default_data_type(p_int64);

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    tiramisu::function function0(name);

    var i(p_int64, "i");
    constant N("N", expr((int64_t) SIZE0), p_int64, true, nullptr, computation::root_dimension, &function0);

    computation S("[N]->{S[i]: 0 <= i < N}", (i * i) - N, true, p_int64, &function0);

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------


    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    function0.allocate_and_map_buffers_automatically();

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    function0.set_arguments({S.get_automatically_allocated_buffer()});
    function0.gen_time_space_domain();
    function0.gen_isl_ast();
    function0.gen_halide_stmt();
    function0.gen_halide_obj("build/generated_fct_test_" + std::string(TEST_NUMBER_STR) + ".o");
}

int main(int argc, char **argv)
{
    generate_function("tiramisu_generated_code");

    return 0;
}
