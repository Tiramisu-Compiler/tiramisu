

#include <isl/set.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>

#include <coli/debug.h>
#include <coli/core.h>

#include <string.h>
#include <Halide.h>
#include "halide_image_io.h"

/* CSR SpMV.
for (i = 0; i < M; i++)
  for (j = row_start[i]; j<row_start[i+1]; j++)
  {
      y[i] += values[j] * x[col_idx[j]];
  }
*/

/* CSR SpMV Simplified.
for (i = 0; i < M; i++)

  int b0 = row_start[i];
  int b1 = row_start[i+1];

  for (j=b0; j<b1; j++)
  {
      int t = col_idx[j];
      y[i] += values[j] * x[t];
  }
*/

#define SIZE0 1000

using namespace coli;

int main(int argc, char **argv)
{
    // Set default coli options.
    global::set_default_coli_options();

    function spmv("spmv");
    buffer b_row_start("b_row_start", 1, {coli::expr(SIZE0)}, p_uint8, NULL, a_input, &spmv);
    buffer b_col_idx("b_col_idx", 1, {coli::expr(SIZE0)}, p_uint8, NULL, a_input, &spmv);
    buffer b_values("b_values", 1, {coli::expr(SIZE0*SIZE0)}, p_uint8, NULL, a_input, &spmv);
    buffer b_x("b_x", 1, {coli::expr(SIZE0*SIZE0)}, p_uint8, NULL, a_input, &spmv);
    buffer b_y("b_y", 1, {coli::expr(SIZE0*SIZE0)}, p_uint8, NULL, a_output, &spmv);

    expr e_M = expr((int32_t) SIZE0);
    constant M("M", &e_M, p_int32, true, NULL, 0, &spmv);

    computation c_row_start("[M]->{c_row_start[i]: 0<=i<M}", NULL, false, p_uint8, &spmv);
    computation c_col_idx("[b0,b1]->{c_col_idx[j]: b0<=j<b1}", NULL, false, p_uint8, &spmv);
    computation c_values("[b0,b1]->{c_values[j]: b0<=j<b1}", NULL, false, p_uint8, &spmv);
    computation c_x("[M,b0,b1]->{c_x[j]: b0<=j<b1}", NULL, false, p_uint8, &spmv);

    computation c_y("[M,b0,b1]->{c_y[i,j]: 0<=i<M and b0<=j<b1}", NULL, true, p_uint8, &spmv);

    spmv.set_context_set("[M,b0,b1]->{: M>0 and b0>0 and b1>0 and b1>b0 and b1%4=0}");

    expr e_t = c_col_idx(idx("j"));
    constant t("t", &e_t, p_int32, false, &c_y, 1, &spmv);
    expr e_b1 = c_row_start(idx("i") + 1);
    constant b1("b1", &e_b1, p_int32, false, &t, 0, &spmv);
    expr e_b0 = c_row_start(idx("i"));
    constant b0("b0", &e_b0, p_int32, false, &b1, 0, &spmv);

    expr e_y = c_y(idx("i")) + c_values(idx("j")) * c_x(idx("t"));
    c_y.set_expression(&e_y);

    c_row_start.set_access("{c_row_start[i]->b_row_start[i]}");
    c_col_idx.set_access("{c_col_idx[j]->b_col_idx[j]}");
    c_values.set_access("{c_values[j]->b_values[j]}");
    c_x.set_access("{c_x[j]->b_x[j]}");
    c_y.set_access("{c_y[i,j]->b_y[i]}");

     b0.set_schedule(      "[M]->{b0[i]->[i,0,0,0]: 0<=i<M}");
     b1.set_schedule("      [M]->{b1[i]->[i,1,0,0]: 0<=i<M}");
      t.set_schedule("[M,b0,b1]->{  t[i,j]->[i,2,j1,1,j2,0]: j1= floor(j/4) and j2 = (j%4) and 0<=i<M and b0<=j<(b1/4) and b1%4=0 and b1>b0 and b1>1 and b0>=1 and b1>=b0+1;   t[i,j]->[i,2,j1,0,j2,0]: j1= floor(j/4) and j2 = (j%4) and 0<=i<M and (b1/4)<=j<b1 and b1>b0 and b1>1 and b0>=1 and b1>=b0+1;}");
    c_y.set_schedule("[M,b0,b1]->{c_y[i,j]->[i,2,j1,1,j2,1]: j1= floor(j/4) and j2 = (j%4) and 0<=i<M and b0<=j<(b1/4) and b1%4=0 and b1>b0 and b1>1 and b0>=1 and b1>=b0+1; c_y[i,j]->[i,2,j1,0,j2,1]: j1= floor(j/4) and j2 = (j%4) and 0<=i<M and (b1/4)<=j<b1 and b1>b0 and b1>1 and b0>=1 and b1>=b0+1;}");

     c_y.tag_vector_dimension(2);
     t.tag_vector_dimension(2);

    /*
    c_y.split(2, 4);
    c_y.tag_vector_dimension(2);
    t.split(2, 4);
    t.tag_vector_dimension(2);
    c_y.tile(0,1,32,32);
    */
    c_y.tag_parallel_dimension(0);

    spmv.set_arguments({&b_row_start, &b_col_idx, &b_values, &b_x, &b_y});

    // Generate code
    spmv.gen_isl_ast();
    spmv.gen_halide_stmt();
    spmv.gen_halide_obj("build/generated_fct_tutorial_04.o");

    // Some debugging
    spmv.dump_halide_stmt();

    return 0;
}
