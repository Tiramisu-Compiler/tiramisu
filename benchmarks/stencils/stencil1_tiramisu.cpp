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

using namespace tiramisu;

#define SIZE0 240
#define SIZE1 240
#define DATA_TYPE p_uint16

void generate_tiramisu_obj_file()
{
    // Set default tiramisu options.
    global::set_default_tiramisu_options();

    function stencil1("stencil1_tiramisu");
    buffer in("in", 1, {tiramisu::expr(SIZE0)}, DATA_TYPE, NULL, a_input, &stencil1);
    buffer b0("b0", 1, {tiramisu::expr(SIZE0)}, DATA_TYPE, NULL, a_temporary, &stencil1);
    buffer b1("b1", 1, {tiramisu::expr(SIZE0)}, DATA_TYPE, NULL, a_output, &stencil1);
    constant N("N", expr((int32_t) SIZE0), p_int32, true, NULL, 0, &stencil1);

    var i = var("i");

    computation c_in("[N]->{c_in[i]: 1<=i<=80-2}", expr(), false, DATA_TYPE, &stencil1);
    computation c_b0("[N]->{c_b0[i]: 1<=i<=80-2}", c_in(i-1) + c_in(i) + c_in(i+1),  true, DATA_TYPE, &stencil1);
    computation c_b1("[N]->{c_b1[i]: 1<=i<=80-2}", c_b0(i-1) + c_b0(i) + c_b0(i+1),  true, DATA_TYPE, &stencil1);

    // Map the computations to a buffer.
    c_in.set_access("{c_in[i]->in[i]}");
    c_b0.set_access("{c_b0[i]->b0[i]}");
    c_b1.set_access("{c_b1[i]->b1[i]}");

    // Set the arguments to stencil1
    stencil1.set_arguments({&in, &b1});

    stencil1.set_context_set("[N]->{: N>1}]");
    c_b0.set_schedule("{c_b0[i]->c_b0[0, 0, i0, 1, i1, 0, 0, 0]: i0=floor(i/4) and i1=i%4; c_b0[i]->c_b0[0, i0, 0, 0, 0, 0, 0]: i0=floor(i/4)and i%4=-1;c_b0[i]->c_b0[0, i0, 2, 0, 0, 0, 0]: i0=floor(i/4) and i%4=1}");
    c_b1.set_schedule("{c_b1[i]->c_b1[0, 0, i0, 3, i1, 1, 0, 0]: i0=floor(i/4) and i1=i%4}");

    // Generate code
    stencil1.gen_time_processor_domain();
    stencil1.gen_isl_ast();
    stencil1.gen_c_code();
    stencil1.gen_halide_stmt();
    stencil1.gen_halide_obj("build/generated_fct_stencil1.o");

    // Some debugging
    stencil1.dump_iteration_domain();
    stencil1.dump_halide_stmt();

    // Dump all the fields of the stencil1 class.
    stencil1.dump(true);
}

int main(int argc, char **argv)
{
    generate_tiramisu_obj_file();

    return 0;
}
