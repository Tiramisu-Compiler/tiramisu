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

using namespace tiramisu;

int main(int argc, char **argv)
{
    // Set default tiramisu options.
    global::set_default_tiramisu_options();

    function waxpby("waxpby");

    constant M_CST("M", expr(N), p_int32, true, NULL, 0, &waxpby);

    // Inputs
    computation x("[M]->{x[j]: 0<=j<M}", tiramisu::expr(), false, p_float64, &waxpby);
    computation y("[M]->{y[j]: 0<=j<M}", tiramisu::expr(), false, p_float64, &waxpby);
    computation alpha("[M]->{alpha[0]}", tiramisu::expr(), false, p_float64, &waxpby);
    computation beta("[M]->{beta[0]}", tiramisu::expr(), false, p_float64, &waxpby);

    tiramisu::var j("j");
    computation w("[M]->{w[j]: 0<=j<M}", alpha(0)*x(j) + beta(0)*y(j), true, p_float64, &waxpby);

    waxpby.set_context_set("[M]->{: M>0}");

    // -----------------------------------------------------------------
    // Layer II
    // ----------------------------------------------------------------- 

    // ---------------------------------------------------------------------------------
    // Layer III
    // ---------------------------------------------------------------------------------
    buffer b_x("b_x", {tiramisu::expr(M)}, p_float64, a_input, &waxpby);
    buffer b_y("b_y", {tiramisu::expr(M)}, p_float64, a_input, &waxpby);
    buffer b_alpha("b_alpha", {tiramisu::expr(0)}, p_float64, a_input, &waxpby);
    buffer b_beta("b_beta", {tiramisu::expr(0)}, p_float64, a_input, &waxpby);
    buffer b_w("b_w", {tiramisu::expr(M)}, p_float64, a_output, &waxpby);

    x.set_access("{x[j]->b_x[j]}");
    y.set_access("{y[j]->b_y[j]}");
    alpha.set_access("{alpha[0]->b_alpha[0]}");
    beta.set_access("{beta[0]->b_beta[0]}");
    w.set_access("{w[j]->b_w[j]}");

    // ------------------------------------------------------------------
    // Generate code
    // ------------------------------------------------------------------
    waxpby.set_arguments({&b_alpha, &b_x, &b_beta, &b_y, &b_w});
    waxpby.gen_time_space_domain();
    waxpby.gen_isl_ast();
    waxpby.gen_halide_stmt();
    waxpby.gen_halide_obj("generated_waxpby.o");

    return 0;
}
