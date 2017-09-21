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

#include "wrapper_test_76.h"

using namespace tiramisu;

/**
 * Checks that different variables with the same name have consistent type.
 */

void generate_function(std::string name)
{
    tiramisu::global::set_default_tiramisu_options();

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    tiramisu::function function0(name);
    constant N("N", expr(DSIZE), global::get_loop_iterator_default_data_type(),
               true, NULL, 0, &function0);
    var i(p_uint16, "i");
    assert(var("i").get_data_type() == p_uint16);
    var other_i("i"), other_i_typed(p_uint16, "i");
    assert(i.is_equal(other_i));
    assert(i.is_equal(other_i_typed));
    assert(other_i.is_equal(other_i_typed));
    computation S("[N] -> {S[j]: 0 <= j < N}",
                  expr((uint16_t) 5), true, p_uint16, &function0);

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    buffer B("B", {DSIZE}, p_uint16, a_output, &function0);
    S.set_access("{S[j] -> B[j]}");

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    function0.set_arguments({&B});
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
