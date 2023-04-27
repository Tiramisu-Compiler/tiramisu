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

#include "wrapper_test_94.h"

using namespace tiramisu;

/**
 * Test PLDI motivating example.
 */

void generate_function(std::string name, int size, int val0)
{
    tiramisu::global::set_default_tiramisu_options();

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    tiramisu::function function0(name);
    tiramisu::constant N("N", tiramisu::expr((int32_t) size), p_int32, true, NULL, 0, &function0);
    tiramisu::var i("i"), j("j"), c("c"), o("o");
    tiramisu::computation S0("[N]->{S0[o,i,j,c]: 0<=o<10000 and 0<=i<N and 0<=j<N and 0<=c<3}", tiramisu::expr((float) val0), true, p_float32, &function0);
    tiramisu::computation S1("[N]->{S1[o,i,j,c]: 0<=o<10000 and 0<=i<N and 0<=j<N and 0<=c<3}", (S0(o,i,j,c) + tiramisu::expr((float)1)), true, p_float32, &function0);

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    S1.after(S0, c);
    S0.interchange(j,c);
    S1.interchange(j,c);
    S0.tag_parallel_level(o);
    S0.tag_vector_level(j, 8);
    S1.tag_vector_level(j, 8);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    tiramisu::buffer buf0("buf0", {SIZE, SIZE, 3}, tiramisu::p_float32, a_temporary, &function0);
    S0.set_access("[N,M]->{S0[o,i,c,j]->buf0[i,j,c]: 0<=i<N and 0<=j<N and 0<=c<3}");
    tiramisu::buffer buf1("buf1", {SIZE, SIZE, 3}, tiramisu::p_float32, a_output, &function0);
    S1.set_access("[N,M]->{S1[o,i,c,j]->buf1[i,j,c]: 0<=i<N and 0<=j<N and 0<=c<3}");

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    function0.set_arguments({&buf1});
    function0.gen_time_space_domain();
    function0.gen_isl_ast();
    function0.gen_halide_stmt();
    function0.gen_halide_obj("generated_fct_test_" + std::string(TEST_NUMBER_STR) + ".o");
}

int main(int argc, char **argv)
{
    generate_function("tiramisu_generated_code", SIZE, 1);

    return 0;
}
