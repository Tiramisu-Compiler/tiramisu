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

#define THREADS 32
#define B0 64
#define B1 8
#define B2 16
#define PARTITIONS (67108864/THREADS)


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
    computation c_spmv("[M,b0,b1]->{c_spmv[i,j]: 0<=i<M and b0<=j<b1}", tiramisu::expr(), true, p_float64, &cg);
    computation c_spmv_wrapper("[M,b0,b1]->{c_spmv_wrapper[i]: 0<=i<M}", tiramisu::expr(), false, p_float64, &cg);


    c_spmv.add_associated_let_stmt("t", c_col_idx(var("c5")));
    constant b1("b1", c_row_start(var("i") + 1), p_int32, false, &c_spmv, 0, &cg);
    constant b0("b0", c_row_start(var("i")), p_int32, false, &b1, 0, &cg);

    expr e_y = c_spmv(var("i"), var("j")) + c_values(var("j")) * w(var("t"));
    c_spmv.set_expression(e_y);

    // dot
    computation res_alloc("[M]->{res_alloc[-10]}", tiramisu::expr(tiramisu::o_allocate, "b_res"), true, p_none, &cg);
    computation  res_init("[M]->{ res_init[t]: 0<=t<(M/"+std::to_string(PARTITIONS)+")}", tiramisu::expr((double) 0), true, p_float64, &cg);
    computation mul_alloc("[M]->{mul_alloc[j]: 0<=j<(M/"+std::to_string(PARTITIONS)+")}", tiramisu::expr(tiramisu::o_allocate, "b_mul"), true, p_float64, &cg);
    computation       mul("[M]->{ mul[j]: 0<=j<M}", c_spmv_wrapper(j)*y(j), true, p_float64, &cg);
    computation       res("[M]->{ res[j]: 0<=j<M}", tiramisu::expr(), true, p_float64, &cg);
    res.set_expression(res(j) + mul(j));
    computation res_global("[M]->{res_global[t]: 0<=t<(M/"+std::to_string(PARTITIONS)+")}", tiramisu::expr(),    true, p_float64, &cg);
    res_global.set_expression(res_global(var("t")) + res_init(var("t")));


    cg.set_context_set("[M,b0,b1]->{: M>0 and M%"+std::to_string(PARTITIONS)+"=0 and b0>0 and b1>0 and b1>b0}");

    // -----------------------------------------------------------------
    // Layer II
    // ----------------------------------------------------------------- 
    w.split(0, PARTITIONS);
    w.tag_parallel_level(0);
    w.split(1, B0);
    w.split(2, B1);
    w.tag_unroll_level(2);
    w.tag_vector_level(3, B1);

    // spmv schedule
    // ----------------------
    b0.split(0, PARTITIONS);
    b1.split(0, PARTITIONS);
    c_spmv.split(0, PARTITIONS);
    c_spmv.tag_parallel_level(0);

 
    // dot schedule
    // ---------------------

    // Split (prepare for parallelization)
    mul.split(0, PARTITIONS);
    res.split(0, PARTITIONS);

    // Split (prepare for vectorization and split)
    mul.split(1, B2);
    res.split(1, B2);

    // Vectorization and unrolling
    mul.tag_vector_level(2, B2);
    res.tag_unroll_level(2);

    // parallelization
    res.tag_parallel_level(0);

    // Ordering
    b0.after_low_level(w, -1);
    b1.after_low_level(b0, 1);
    c_spmv.after_low_level(b1,2);

    res_init.after_low_level(c_spmv, -1);
    mul_alloc.after_low_level(res_init,1);
    mul.after_low_level(mul_alloc,1);
    res.after_low_level(mul, 1);
    res_global.after_low_level(res, -1);

    // ---------------------------------------------------------------------------------
    // Layer III
    // ---------------------------------------------------------------------------------
    // waxpby
    buffer b_SIZES("b_SIZES", {tiramisu::expr(1)}, p_int32, a_input, &cg);
    buffer b_x("b_x", {tiramisu::var("M")}, p_float64, a_input, &cg);
    buffer b_y("b_y", {tiramisu::var("M")}, p_float64, a_input, &cg);
    buffer b_alpha("b_alpha", {tiramisu::expr(1)}, p_float64, a_input, &cg);
    buffer b_beta("b_beta", {tiramisu::expr(1)}, p_float64, a_input, &cg);
    buffer b_w("b_w", {tiramisu::var("M")}, p_float64, a_output, &cg);

    SIZES.set_access("{SIZES[0]->b_SIZES[0]}");
    x.set_access("{x[j]->b_x[j]}");
    y.set_access("{y[j]->b_y[j]}");
    alpha.set_access("{alpha[0]->b_alpha[0]}");
    beta.set_access("{beta[0]->b_beta[0]}");
    w.set_access("{w[j]->b_w[j]}");

    // spmv
    buffer b_row_start("b_row_start", {tiramisu::expr(N)}, p_int32, a_input, &cg);
    buffer b_col_idx("b_col_idx", {tiramisu::expr(N)}, p_int32, a_input, &cg);
    buffer b_values("b_values", {tiramisu::expr((N*N))}, p_float64, a_input, &cg);
    buffer b_spmv("b_spmv", {tiramisu::expr(N*N)}, p_float64, a_output, &cg);

    c_row_start.set_access("{c_row_start[i]->b_row_start[i]}");
    c_col_idx.set_access("{c_col_idx[j]->b_col_idx[j]}");
    c_values.set_access("{c_values[j]->b_values[j]}");
    c_spmv.set_access("{c_spmv[i,j]->b_spmv[i]}");
    c_spmv_wrapper.set_access("{c_spmv_wrapper[i]->b_spmv[i]}");

    // dot
    buffer b_mul("b_mul", {tiramisu::expr((int) B2)}, p_float64, a_temporary, &cg);
    b_mul.set_auto_allocate(false);
    buffer b_res("b_res", {tiramisu::var("M")/tiramisu::expr((int) PARTITIONS)}, p_float64, a_temporary, &cg);
    b_res.set_auto_allocate(false);
    buffer b_res_global("b_res_global", {tiramisu::expr((int) 1)}, p_float64, a_output, &cg);

    mul.set_access("{mul[j]->b_mul[j%"+std::to_string(B2)+"]}");
    res_global.set_access("{res_global[j]->b_res_global[0]}");
    res_init.set_access("{res_init[t]->b_res[t]}");
    res.set_access("{res[j]->b_res[j/"+std::to_string(PARTITIONS)+"]}");



    // ------------------------------------------------------------------
    // Generate code
    // ------------------------------------------------------------------
    cg.set_arguments({&b_SIZES, &b_alpha, &b_x, &b_beta, &b_y, &b_w, &b_row_start, &b_col_idx, &b_values, &b_spmv, &b_res_global});
    cg.gen_time_space_domain();
    cg.gen_isl_ast();
    cg.gen_halide_stmt();
    cg.gen_halide_obj("generated_cg.o");

    return 0;
}
