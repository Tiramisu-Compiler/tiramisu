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

#include "wrapper_test_89.h"

using namespace tiramisu;

/**
 * Test buffer with dynamic size. 
 */

void generate_function(std::string name, int val0)
{
    tiramisu::global::set_default_tiramisu_options();
    

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    tiramisu::function function0(name);
    tiramisu::var i("i"), j("j"), io("io"), jo("jo"), ii("ii"), ji("ji");

    tiramisu::computation SIZEs_computation("{SIZEs_computation[0]}", tiramisu::expr(), false, p_int32, &function0);
    tiramisu::constant			  N("N"			    , SIZEs_computation(0), p_int32, true, NULL, 0, &function0);
    tiramisu::computation		 S0("[N]->{S0[i,j]: 0<=i<N and 0<=j<N}", tiramisu::expr((uint8_t) val0), true, p_uint8, &function0);
    tiramisu::computation		 S1("[N]->{S1[i,j]: 0<=i<N and 0<=j<N}", S0(i,j), true, p_uint8, &function0);

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    function0.add_context_constraints("[N]->{: N>32}");
    S0.tile(i, j, 8, 8, io, jo, ii, ji);
    S0.vectorize(ji, 8);
    S1.tile(i,j, 32, 32, io, jo, ii, ji);
    S1.after(S0.get_last_update(), computation::root);
    S1.vectorize(ji, 8);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    tiramisu::buffer SIZEs_buffer("SIZEs_buffer", {1}				  , tiramisu::p_int32, a_input, &function0);
    tiramisu::buffer	     S0_b("S0_b", {tiramisu::var("N"), tiramisu::var("N")}, tiramisu::p_uint8, a_temporary, &function0);
    tiramisu::buffer	     S1_b("S1_b", {tiramisu::var("N"), tiramisu::var("N")}, tiramisu::p_uint8, a_output, &function0);

    SIZEs_computation.store_in(&SIZEs_buffer);
    S0.store_in(&S0_b);
    S1.store_in(&S1_b);

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    function0.set_arguments({&SIZEs_buffer, &S1_b});
    function0.gen_time_space_domain();
    function0.gen_isl_ast();
    function0.gen_halide_stmt();
    function0.gen_halide_obj("generated_fct_test_" + std::string(TEST_NUMBER_STR) + ".o");
}

int main(int argc, char **argv)
{
    generate_function("tiramisu_generated_code", 5);

    return 0;
}
