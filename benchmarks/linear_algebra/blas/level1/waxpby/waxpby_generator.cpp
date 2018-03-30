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

/* waxpby.

inputs:
--------
- x[]
- y[]
- alpha
- beta

for (int i=0; i<n; i++)
	w[i] = alpha * x[i] + beta * y[i];

*/

#define nrow SIZE

#define THREADS 32
#define B0 256
#define B1 8

#define B0s std::to_string(B0)
#define B1s std::to_string(B1)

#define EXTRA_OPTIMIZATIONS 1


using namespace tiramisu;

int main(int argc, char **argv)
{
    // Set default tiramisu options.
    global::set_default_tiramisu_options();

    function waxpby("waxpby");


    // Inputs
    computation SIZES("[M]->{SIZES[0]}", tiramisu::expr(), false, p_int32, &waxpby);
    computation x("[M]->{x[j]: 0<=j<M}", tiramisu::expr(), false, p_float64, &waxpby);
    computation y("[M]->{y[j]: 0<=j<M}", tiramisu::expr(), false, p_float64, &waxpby);
    computation alpha("[M]->{alpha[0]}", tiramisu::expr(), false, p_float64, &waxpby);
    computation beta("[M]->{beta[0]}", tiramisu::expr(), false, p_float64, &waxpby);

    constant M_CST("M", SIZES(0), p_int32, true, NULL, 0, &waxpby);

    tiramisu::var j("j");
    computation w("[M]->{w[j]: 0<=j<M}", alpha(0)*x(j) + beta(0)*y(j), true, p_float64, &waxpby);

    waxpby.set_context_set("[M]->{: M>0 and M%"+B0s+"=0}");

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
    buffer b_SIZES("b_SIZES", {tiramisu::expr(1)}, p_int32, a_input, &waxpby);
    buffer b_x("b_x", {tiramisu::expr(nrow)}, p_float64, a_input, &waxpby);
    buffer b_y("b_y", {tiramisu::expr(nrow)}, p_float64, a_input, &waxpby);
    buffer b_alpha("b_alpha", {tiramisu::expr(1)}, p_float64, a_input, &waxpby);
    buffer b_beta("b_beta", {tiramisu::expr(1)}, p_float64, a_input, &waxpby);
    buffer b_w("b_w", {tiramisu::expr(nrow)}, p_float64, a_output, &waxpby);

    SIZES.set_access("{SIZES[0]->b_SIZES[0]}");
    x.set_access("{x[j]->b_x[j]}");
    y.set_access("{y[j]->b_y[j]}");
    alpha.set_access("{alpha[0]->b_alpha[0]}");
    beta.set_access("{beta[0]->b_beta[0]}");
    w.set_access("{w[j]->b_w[j]}");

    // ------------------------------------------------------------------
    // Generate code
    // ------------------------------------------------------------------
    waxpby.set_arguments({&b_SIZES, &b_alpha, &b_x, &b_beta, &b_y, &b_w});
    waxpby.gen_time_space_domain();
    waxpby.gen_isl_ast();
    waxpby.gen_halide_stmt();
    waxpby.gen_halide_obj("generated_waxpby.o");

    return 0;
}
