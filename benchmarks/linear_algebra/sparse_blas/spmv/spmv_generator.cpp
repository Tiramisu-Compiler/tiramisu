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

/* CSR SpMV.
for (i = 0; i < M; i++)
    for (j = row_start[i]; j<row_start[i+1]; j++)
    {
        y[i] += values[j] * x[col_idx[j]];
    }
*/

/* CSR SpMV Simplified.

inputs:
--------
- row_start[]
- col_idx[]
- values[]
- y[]
- x[]

for (i = 0; i < M; i++)
    int b0 = row_start[i];
    int b1 = row_start[i+1];

    for (j=b0; j<b1; j++)
    {
        int t = col_idx[j];
        y[i] += values[j] * x[t];
    }
*/


using namespace tiramisu;

int main(int argc, char **argv)
{
    // Set default tiramisu options.
    global::set_default_tiramisu_options();

    function spmv("spmv");

    constant M_CST("M", expr(N), p_int32, true, NULL, 0, &spmv);

    // Inputs
    computation c_row_start("[M]->{c_row_start[i]: 0<=i<M}", tiramisu::expr(), false, p_uint8, &spmv);
    computation c_col_idx("[b0,b1]->{c_col_idx[j]: b0<=j<b1}", tiramisu::expr(), false, p_uint8, &spmv);
    computation c_values("[b0,b1]->{c_values[j]: b0<=j<b1}", tiramisu::expr(), false, p_uint8, &spmv);
    computation c_x("[M,b0,b1]->{c_x[j]: b0<=j<b1}", tiramisu::expr(), false, p_uint8, &spmv);


    computation c_y("[M,b0,b1]->{c_y[i,j]: 0<=i<M and b0<=j<b1}", tiramisu::expr(), true, p_uint8, &spmv);

    constant t("t", c_col_idx(var("j")), p_int32, false, &c_y, 1, &spmv);
    constant b1("b1", c_row_start(var("i") + 1), p_int32, false, &t, 0, &spmv);
    constant b0("b0", c_row_start(var("i")), p_int32, false, &b1, 0, &spmv);

    expr e_y = c_y(var("i"), var("j")) + c_values(var("j")) * c_x(var("t"));
    c_y.set_expression(e_y);

    spmv.set_context_set("[M,b0,b1]->{: M>0 and b0>0 and b1>0 and b1>b0}");

    // -----------------------------------------------------------------
    // Layer II
    // ----------------------------------------------------------------- 
    b1.after_low_level(b0, 0);
    t.after_low_level(b1,0);
    c_y.after_low_level(t,1);


    //c_y.tag_parallel_level(0);

    // ---------------------------------------------------------------------------------
    // Layer III
    // ---------------------------------------------------------------------------------
    buffer b_row_start("b_row_start", {tiramisu::expr(N)}, p_int32, a_input, &spmv);
    buffer b_col_idx("b_col_idx", {tiramisu::expr(N)}, p_int32, a_input, &spmv);
    buffer b_values("b_values", {tiramisu::expr((N*N))}, p_float64, a_input, &spmv);
    buffer b_x("b_x", {tiramisu::expr(N*N)}, p_float64, a_input, &spmv);
    buffer b_y("b_y", {tiramisu::expr(N*N)}, p_float64, a_output, &spmv);

    c_row_start.set_access("{c_row_start[i]->b_row_start[i]}");
    c_col_idx.set_access("{c_col_idx[j]->b_col_idx[j]}");
    c_values.set_access("{c_values[j]->b_values[j]}");
    c_x.set_access("{c_x[j]->b_x[j]}");
    c_y.set_access("{c_y[i,j]->b_y[i]}");

    // ------------------------------------------------------------------
    // Generate code
    // ------------------------------------------------------------------
    spmv.set_arguments({&b_row_start, &b_col_idx, &b_values, &b_x, &b_y});
    spmv.gen_time_space_domain();
    spmv.gen_isl_ast();
    spmv.gen_halide_stmt();
    spmv.gen_halide_obj("generated_spmv.o");

    return 0;
}
