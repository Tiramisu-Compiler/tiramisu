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

#include "wrapper_test_91.h"

using namespace tiramisu;

/**
 * New update/reduction formulation.
 * for i = 0, N
 *   for j = 0, N
 *     for k = 0, N
 *       f(j,k) = 0;
 *     g(i,j) = 0;
 * for i = 0, N
 *   for j = 0, N
 *     for k0 = 0, N
 *       for k1 = 0, N
 *         f(k0,k1) += 1;
 *	   g(i,j) = f(0,0);
 *
 *
 * The expected Halide IR generated code is
 *
 * allocate f_b[uint8 * N * N]
 * for (c1, 0, N) {
 *   for (c3, 0, N) {
 *     for (c5, 0, N) {
 *       f_b[(c5 + (c3*N))] = (uint8)0
 *     }
 *     g_b[(c3 + (c1*N))] = (uint8)0
 *   }
 * }
 * for (c1, 0, N) {
 *   for (c3, 0, N) {
 *     for (c5, 0, N) {
 *       for (c7, 0, N) {
 *         f_b[(c7 + (c5*N))] = (f_b[(c7 + (c5*N))] + (uint8)1)
 *         g_b[(c7 + (c5*N))] = f_b[0]
 *       }
 *     }
 *   }
 * }
 * free f_b
 */

void generate_function(std::string name, int val0)
{
    tiramisu::global::set_default_tiramisu_options();


    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    tiramisu::function function0(name);

    tiramisu::computation SIZEs_computation("{SIZEs_computation[0]}", tiramisu::expr(), false, p_int32, &function0);
    tiramisu::constant			  N("N"			    , SIZEs_computation(0), p_int32, true, NULL, 0, &function0);
    tiramisu::computation		 f("[N]->{f[i,j,k]: 0<=i<N and 0<=j<N and 0<=k<N}", tiramisu::expr((uint8_t) 0), true, p_uint8, &function0);
    tiramisu::computation		 g("[N]->{g[i,j]: 0<=i<N and 0<=j<N}", tiramisu::expr((uint8_t) 0), true, p_uint8, &function0);
    tiramisu::computation		 f_1("[N]->{f_1[i,j,k0,k1]: 0<=i<N and 0<=j<N and 0<=k0<N and 0<=k1<N}", tiramisu::expr(), true, p_uint8, &function0);
    f_1.set_expression(f_1(tiramisu::var("i"), tiramisu::var("j"), tiramisu::var("k0"), tiramisu::var("k1")) + tiramisu::expr((uint8_t) 1));
    tiramisu::computation		 g_1("[N]->{g_1[i,j,k0,k1]: 0<=i<N and 0<=j<N and 0<=k0<N and 0<=k1<N}", tiramisu::expr(), true, p_uint8, &function0);
    g_1.set_expression(f_1(tiramisu::var("i"), tiramisu::var("j"), 0, 0));//tiramisu::var("k0"),  tiramisu::var("k1")));

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    g.after(f, tiramisu::var("j"));
    f_1.after(g, computation::root);
    g_1.after(f_1, tiramisu::var("k1"));

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    tiramisu::buffer SIZEs_buffer("SIZEs_buffer", {1}				  , tiramisu::p_int32, a_input, &function0);
    tiramisu::buffer	     f_b("f_b", {tiramisu::var("N"), tiramisu::var("N")}, tiramisu::p_uint8, a_temporary, &function0);
    tiramisu::buffer	     g_b("g_b", {tiramisu::var("N"), tiramisu::var("N")}, tiramisu::p_uint8, a_output, &function0);

    SIZEs_computation.store_in(&SIZEs_buffer);
    f.set_access("{f[i,j,k]->f_b[j,k]}");
    g.set_access("{g[i,j]->g_b[i,j]}");
    f_1.set_access("{f_1[i,j,k0,k1]->f_b[k0,k1]}");
    g_1.set_access("{g_1[i,j,k0,k1]->g_b[k0,k1]}");

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    function0.set_arguments({&SIZEs_buffer, &g_b});
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
