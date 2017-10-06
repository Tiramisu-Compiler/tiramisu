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

#include "wrapper_test_90.h"

using namespace tiramisu;

/**
 * Test inserting new statments between already existing statements.
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
    tiramisu::computation		 S2("[N]->{S2[i,j]: 0<=i<N and 0<=j<N}", S1(i,j), true, p_uint8, &function0);
    tiramisu::computation		 S3("[N]->{S3[i,j]: 0<=i<N and 0<=j<N}", S2(i,j), true, p_uint8, &function0);


    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    /**
      *  for (i ...)
      *	    for (j ...)
      *		S0;
      *	 for (i ...)
      *	    S1;
      *	    for (j ...)
      *		S2;
      *	 for (i ...)
      *	    A;
      *	    for (j ...)
      *		S3;
      */

    S2.after(S0, computation::root);
    S1.compute_at(S2, tiramisu::var("i"));
    S3.after(S2, computation::root);
    tiramisu::buffer S3_b("S3_b", {tiramisu::var("N"), tiramisu::var("N")}, tiramisu::p_uint8, a_output, &function0);
    tiramisu::computation *A = S3_b.allocate_at(S3, 1);
    A->before(S3, tiramisu::var("i"));

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    tiramisu::buffer SIZEs_buffer("SIZEs_buffer", {1}				  , tiramisu::p_int32, a_input, &function0);
    tiramisu::buffer	     S0_b("S0_b", {tiramisu::var("N"), tiramisu::var("N")}, tiramisu::p_uint8, a_temporary, &function0);
    tiramisu::buffer	     S1_b("S1_b", {tiramisu::var("N"), tiramisu::var("N")}, tiramisu::p_uint8, a_temporary, &function0);
    tiramisu::buffer	     S2_b("S2_b", {tiramisu::var("N"), tiramisu::var("N")}, tiramisu::p_uint8, a_temporary, &function0);

    SIZEs_computation.bind_to(&SIZEs_buffer);
    S0.bind_to(&S0_b);
    S1.bind_to(&S1_b);
    S2.bind_to(&S2_b);
    S3.bind_to(&S3_b);

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    function0.set_arguments({&SIZEs_buffer, &S3_b});
    function0.gen_time_space_domain();
    function0.gen_isl_ast();
    function0.gen_halide_stmt();
    function0.gen_halide_obj("build/generated_fct_test_" + std::string(TEST_NUMBER_STR) + ".o");
}

int main(int argc, char **argv)
{
    generate_function("tiramisu_generated_code", 5);

    return 0;
}
