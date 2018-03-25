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

#define nrow SIZE

using namespace tiramisu;

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

    constant M_CST("M", expr(nrow), p_int32, true, NULL, 0, &cg);

    // Inputs
    computation x("[M]->{x[j]: 0<=j<M}", tiramisu::expr(), false, p_float64, &cg);
    computation y("[M]->{y[j]: 0<=j<M}", tiramisu::expr(), false, p_float64, &cg);
    computation beta("[M]->{beta[0]}", tiramisu::expr(), false, p_float64, &cg);

    // Algorithm
    tiramisu::var j("j");
    computation w("[M]->{w[j]: 0<=j<M}", x(j) + beta(0)*y(j), true, p_float64, &cg);

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
    buffer b_r("b_r", {tiramisu::expr(nrow)}, p_float64, a_input, &cg);
    buffer b_p("b_p", {tiramisu::expr(nrow)}, p_float64, a_input, &cg);
    buffer b_beta("b_beta", {tiramisu::expr(0)}, p_float64, a_input, &cg);

    x.set_access("{x[j]->b_r[j]}");
    y.set_access("{y[j]->b_p[j]}");
    beta.set_access("{beta[0]->b_beta[0]}");
    w.set_access("{w[j]->b_p[j]}");

    // ------------------------------------------------------------------
    // Generate code
    // ------------------------------------------------------------------
    cg.set_arguments({&b_r, &b_beta, &b_p});
    cg.gen_time_space_domain();
    cg.gen_isl_ast();
    cg.gen_halide_stmt();
    cg.gen_halide_obj("generated_cg.o");

    return 0;
}
