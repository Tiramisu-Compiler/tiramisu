#include <isl/set.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>

#include <tiramisu/debug.h>
#include <tiramisu/core.h>
#include "benchmarks.h"

#include <string.h>
#include <Halide.h>

/* cg.

inputs:
--------
- x[]
- y[]
- beta

Algorithm:
-----------
// waxpby (nrow, 1.0, r, beta, p, p);
for (int i=0; i<nrow; i++)
	p[i] = r[i] + beta * p[i];

*/

using namespace tiramisu;

#define nrow SIZE

#define THREADS 32
#define B0 256
#define B1 8
#define B0s std::to_string(B0)
#define B1s std::to_string(B1)

#define EXTRA_OPTIMIZATIONS 1

int main(int argc, char **argv)
{
    // Set default tiramisu options.
    global::set_default_tiramisu_options();

    function cg("cg");


    // Inputs
    computation SIZES("[M]->{SIZES[0]}", tiramisu::expr(), false, p_int32, &cg);
    computation x("[M]->{x[j]: 0<=j<M}", tiramisu::expr(), false, p_float64, &cg);
    computation y("[M]->{y[j]: 0<=j<M}", tiramisu::expr(), false, p_float64, &cg);
    computation alpha("[M]->{alpha[0]}", tiramisu::expr(), false, p_float64, &cg);
    computation beta("[M]->{beta[0]}", tiramisu::expr(), false, p_float64, &cg);

    constant M_CST("M", SIZES(0), p_int32, true, NULL, 0, &cg);

    tiramisu::var j("j");
    computation w("[M]->{w[j]: 0<=j<M}", alpha(0)*x(j) + beta(0)*y(j), true, p_float64, &cg);

    cg.set_context_set("[M]->{: M>0 and M%"+B0s+"=0}");

    // -----------------------------------------------------------------
    // Layer II
    // ----------------------------------------------------------------- 
    w.split(0, SIZE/THREADS);
    w.tag_parallel_level(0);
#if EXTRA_OPTIMIZATIONS
    w.split(1, B0);
    w.split(2, B1);
    w.tag_unroll_level(2);
    w.tag_vector_level(3, B1);
#endif

    // ---------------------------------------------------------------------------------
    // Layer III
    // ---------------------------------------------------------------------------------
    buffer b_SIZES("b_SIZES", {tiramisu::expr(1)}, p_int32, a_input, &cg);
    buffer b_x("b_x", {tiramisu::expr(nrow)}, p_float64, a_input, &cg);
    buffer b_y("b_y", {tiramisu::expr(nrow)}, p_float64, a_input, &cg);
    buffer b_alpha("b_alpha", {tiramisu::expr(1)}, p_float64, a_input, &cg);
    buffer b_beta("b_beta", {tiramisu::expr(1)}, p_float64, a_input, &cg);
    buffer b_w("b_w", {tiramisu::expr(nrow)}, p_float64, a_output, &cg);

    SIZES.set_access("{SIZES[0]->b_SIZES[0]}");
    x.set_access("{x[j]->b_x[j]}");
    y.set_access("{y[j]->b_y[j]}");
    alpha.set_access("{alpha[0]->b_alpha[0]}");
    beta.set_access("{beta[0]->b_beta[0]}");
    w.set_access("{w[j]->b_w[j]}");

    // ------------------------------------------------------------------
    // Generate code
    // ------------------------------------------------------------------
    cg.set_arguments({&b_SIZES, &b_alpha, &b_x, &b_beta, &b_y, &b_w});
    cg.gen_time_space_domain();
    cg.gen_isl_ast();
    cg.gen_halide_stmt();
    cg.gen_halide_obj("generated_cg.o");

    return 0;
}
