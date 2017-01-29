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
#include "halide_image_io.h"

/* CSR SpMV Simplified.
for (i = 0; i < M; i++)
  S0(i) = 7;
  S1(i) = 7;
  for (j=0; j<N; j++)
      S2(i,j) = 7;
  S3(i) = 7;
*/

#define SIZE0 10

using namespace tiramisu;

int main(int argc, char **argv)
{
    // Set default tiramisu options.
    global::set_default_tiramisu_options();

    function sequence("sequence");
    buffer b0("b0", 1, {tiramisu::expr(SIZE0)}, p_uint8, NULL, a_output, &sequence);
    buffer b1("b1", 1, {tiramisu::expr(SIZE0)}, p_uint8, NULL, a_output, &sequence);
    buffer b2("b2", 2, {tiramisu::expr(SIZE0),tiramisu::expr(SIZE0)}, p_uint8, NULL, a_output, &sequence);
    buffer b3("b3", 1, {tiramisu::expr(SIZE0)}, p_uint8, NULL, a_output, &sequence);

    expr e_M = expr((int32_t) SIZE0);
    constant M("M", e_M, p_int32, true, NULL, 0, &sequence);

    expr val = tiramisu::expr(1);
    computation c0("[M]->{c0[i]: 0<=i<M}", val, true, p_uint8, &sequence);
    computation c1("[M]->{c1[i]: 0<=i<M}", val, true, p_uint8, &sequence);
    computation c2("[M]->{c2[i,j]: 0<=i<M and 0<=j<M}", val, true, p_uint8, &sequence);
    computation c3("[M]->{c3[i]: 0<=i<M}", val, true, p_uint8, &sequence);

    c0.set_access("{c0[i]->b0[i]}");
    c1.set_access("{c1[i]->b1[i]}");
    c2.set_access("{c2[i,j]->b2[i,j]}");
    c3.set_access("{c3[i]->b3[i]}");

    c0.first(1);
    c1.after(c0, 1);
    c2.after(c1, 1);
    c3.after(c2, 1);

    sequence.set_arguments({&b0, &b1, &b2, &b3});

    // Generate code
    sequence.gen_time_processor_domain();
    sequence.gen_isl_ast();
    sequence.gen_halide_stmt();
    sequence.gen_halide_obj("build/generated_fct_tutorial_05.o");

    // Some debugging
    sequence.dump(true);
    sequence.dump_halide_stmt();

    return 0;
}
