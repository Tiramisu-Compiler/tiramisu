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

#include "wrapper_test_79.h"

using namespace tiramisu;

/**
 * tests replacement
 */

void generate_function(std::string name)
{
    tiramisu::global::set_default_tiramisu_options();
    

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    tiramisu::function function0(name);
    var i("i"), j("j"), k("k");
    constant N("N", expr((int32_t)SIZE0), p_int32, true, NULL, 0, &function0);
    computation Si("{Si[i, j, k]}", i + j + k, false, p_int32, &function0);
    Si.set_inline();
    computation S("[N] -> {S[i,j]: 0 <= i < N and 0 <= j < N}",
                  Si(i*i, 2*i*j, j*j), true, p_int32, &function0);

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    // Nothing to do

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
