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
#define PARTITIONS (SIZE/THREADS)
#define B0s std::to_string(B0)
#define B1s std::to_string(B1)

#define EXTRA_OPTIMIZATIONS 0

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
    computation w("[M]->{w[j]: 0<=j<M}", x(j) + beta(0)*y(j), true, p_float64, &cg);

    computation c_row_start("[M]->{c_row_start[i]: 0<=i<M}", tiramisu::expr(), false, p_int32, &cg);
    computation c_col_idx("[b0,b1]->{c_col_idx[j]: b0<=j<b1}", tiramisu::expr(), false, p_int32, &cg);
    computation c_values("[b0,b1]->{c_values[j]: b0<=j<b1}", tiramisu::expr(), false, p_float64, &cg);
    computation c_y("[M,b0,b1]->{c_y[i,j]: 0<=i<M and b0<=j<b1}", tiramisu::expr(), true, p_float64, &cg);

    constant t("t", c_col_idx(var("j")), p_int32, false, &c_y, 1, &cg);
    constant b1("b1", c_row_start(var("i") + 1), p_int32, false, &t, 0, &cg);
    constant b0("b0", c_row_start(var("i")), p_int32, false, &b1, 0, &cg);

    expr e_y = c_y(var("i"), var("j")) + c_values(var("j")) * w(var("t"));
    c_y.set_expression(e_y);

    cg.set_context_set("[M,b0,b1]->{: M>0 and M%"+B0s+"=0 and b0>0 and b1>0 and b1>b0}");

    // -----------------------------------------------------------------
    // Layer II
    // ----------------------------------------------------------------- 
    w.split(0, PARTITIONS);
    w.tag_parallel_level(0);
#if EXTRA_OPTIMIZATIONS
    w.split(1, B0);
    w.split(2, B1);
    w.tag_unroll_level(2);
    w.tag_vector_level(3, B1);
#endif

    // spmv schedule
    b0.split(0, PARTITIONS);
    b1.split(0, PARTITIONS);
    t.split(0, PARTITIONS);
    c_y.split(0, PARTITIONS);

    c_y.tag_parallel_level(0);


    b0.after_low_level(w, -1);

    b1.after_low_level(b0, 1);
    t.after_low_level(b1,1);
    c_y.after_low_level(t,2);

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

    buffer b_row_start("b_row_start", {tiramisu::expr(N)}, p_int32, a_input, &cg);
    buffer b_col_idx("b_col_idx", {tiramisu::expr(N)}, p_int32, a_input, &cg);
    buffer b_values("b_values", {tiramisu::expr((N*N))}, p_float64, a_input, &cg);
    buffer b_spmv("b_spmv", {tiramisu::expr(N*N)}, p_float64, a_output, &cg);

    c_row_start.set_access("{c_row_start[i]->b_row_start[i]}");
    c_col_idx.set_access("{c_col_idx[j]->b_col_idx[j]}");
    c_values.set_access("{c_values[j]->b_values[j]}");
    c_y.set_access("{c_y[i,j]->b_spmv[i]}");


    // ------------------------------------------------------------------
    // Generate code
    // ------------------------------------------------------------------
    cg.set_arguments({&b_SIZES, &b_alpha, &b_x, &b_beta, &b_y, &b_w, &b_row_start, &b_col_idx, &b_values, &b_spmv});
    cg.gen_time_space_domain();
    cg.gen_isl_ast();
    cg.gen_halide_stmt();
    cg.gen_halide_obj("generated_cg.o");

    return 0;
}
