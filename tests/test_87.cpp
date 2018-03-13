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

#include "wrapper_test_87.h"

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
    tiramisu::computation N_input("{N_input[0]}", tiramisu::expr(), false, p_int32, &function0);
    tiramisu::constant N("N", N_input(0), p_int32, true, NULL, 0, &function0);
    tiramisu::var i("i"), j("j");
    tiramisu::computation S0("[N]->{S0[i,j]: 0<=i<N and 0<=j<N}", tiramisu::expr((uint8_t) val0), true, p_uint8, &function0);
    tiramisu::computation S1("[N]->{S1[i,j]: 0<=i<N and 0<=j<N}", S0(i,j), true, p_uint8, &function0);

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    S1.after(S0, computation::root);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    tiramisu::buffer N_input_b("N_input_b", {1}, tiramisu::p_int32, a_input, &function0);
    N_input.bind_to(&N_input_b);
    tiramisu::buffer S0_b("S0_b", {N_input(0), N_input(0)}, tiramisu::p_uint8, a_temporary, &function0);
    S0.bind_to(&S0_b);
    tiramisu::buffer S1_b("S1_b", {tiramisu::var("N"), tiramisu::var("N")}, tiramisu::p_uint8, a_output, &function0);
    S1.bind_to(&S1_b);

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    function0.set_arguments({&N_input_b, &S1_b});
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
