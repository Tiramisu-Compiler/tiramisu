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

#include "wrapper_test_92.h"

using namespace tiramisu;

/**
 * Allocate an array with the size being the iteraotr of the loop.
 * for i = 0, N
 *   allocate f[i+1];
 *   for j = 0, N
 *       f[i] = 0;
 *       g[i] = f[i];
 *
 */

void generate_function(std::string name, int val0)
{
    tiramisu::global::set_default_tiramisu_options();

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    tiramisu::function function0(name);

    tiramisu::computation SIZEs_computation("{SIZEs_computation[0]}", tiramisu::expr(), false, p_int32, &function0);
    tiramisu::constant			 N("N"			    , SIZEs_computation(0), p_int32, true, NULL, 0, &function0);
    tiramisu::computation		 f("[N]->{f[i, j]: 0<=i<N and 0<=j<N}", tiramisu::expr((uint8_t) val0), true, p_uint8, &function0);
    tiramisu::computation		 g("[N]->{g[i, j]: 0<=i<N and 0<=j<N}", f(tiramisu::var("i"), 0), true, p_uint8, &function0);

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    g.after(f, tiramisu::var("j"));

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    tiramisu::buffer SIZEs_buffer("SIZEs_buffer", {1}				  , tiramisu::p_int32, a_input, &function0);
    tiramisu::buffer	     f_b("f_b", {tiramisu::var("i") + 1}, tiramisu::p_uint8, a_temporary, &function0);
    tiramisu::buffer	     g_b("g_b", {tiramisu::var("N")}, tiramisu::p_uint8, a_output, &function0);
    tiramisu::computation *allocation = f_b.allocate_at(f, tiramisu::var("i"));

    f.after((*allocation), tiramisu::var("i"));

    SIZEs_computation.bind_to(&SIZEs_buffer);
    f.set_access("{f[i,j]->f_b[i]}");
    g.set_access("{g[i,j]->g_b[i]}");

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    function0.set_arguments({&SIZEs_buffer, &g_b});
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
