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

#include "wrapper_test_40.h"

using namespace tiramisu;

/**
 * Test .storage_fold().
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
    tiramisu::var i("i");
    tiramisu::var j("j");
    tiramisu::computation S0("[N]->{S0[i,j]: 0<=i<N and 0<=j<N}", tiramisu::expr((uint8_t) val0), true,
                             p_uint8, &function0);


    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------



    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    tiramisu::buffer buf0("buf0", {10, 10}, tiramisu::p_uint8, a_output, &function0);
    S0.set_access("[N,M]->{S0[i,j]->buf0[i,j]: 0<=i<N and 0<=j<N}");
    S0.storage_fold(i, 2);
    S0.storage_fold(j, 3);

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    function0.set_arguments({&buf0});
    function0.gen_time_space_domain();
    function0.gen_isl_ast();
    function0.gen_halide_stmt();
    function0.gen_halide_obj("build/generated_fct_test_" + std::string(TEST_NUMBER_STR) + ".o");
}

int main(int argc, char **argv)
{
    generate_function("tiramisu_generated_code", SIZE1, 2);

    return 0;
}
