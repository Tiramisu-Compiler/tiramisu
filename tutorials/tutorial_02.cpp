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

/* Halide code.
Func blurxy(Func input, Func blur_y) {
    Func blur_x;
    Var x, y, xi, yi;

    // The algorithm - no storage or order
    blur_x(x, y) = (input(x-1, y) + input(x, y) + input(x+1, y))/3;
    blur_y(x, y) = (blur_x(x, y-1) + blur_x(x, y) + blur_x(x, y+1))/3;

    // The schedule - defines order, locality; implies storage
    blur_y.tile(x, y, xi, yi, 256, 32)
        .vectorize(xi, 8).parallel(y);
    blur_x.compute_at(blur_y, x).vectorize(x, 8);
  }
*/

#define SIZE0 1280
#define SIZE1 768

using namespace tiramisu;

int main(int argc, char **argv)
{
    // Set default tiramisu options.
    global::set_default_tiramisu_options();

    /*
     * Declare a function blurxy.
     * Declare two arguments (tiramisu buffers) for the function: b_input and b_blury
     * Declare an invariant for the function.
     */
    function blurxy("blurxy");
    buffer b_input("b_input", 2, {tiramisu::expr(SIZE0),tiramisu::expr(SIZE1)}, p_uint8, NULL, a_input, &blurxy);
    buffer b_blury("b_blury", 2, {tiramisu::expr(SIZE0),tiramisu::expr(SIZE1)}, p_uint8, NULL, a_output, &blurxy);
    expr e_p0 = expr((int32_t) SIZE0);
    expr e_p1 = expr((int32_t) SIZE1);
    constant p0("N", e_p0, p_int32, true, NULL, 0, &blurxy);
    constant p1("M", e_p1, p_int32, true, NULL, 0, &blurxy);

    // Declare the computations c_blurx and c_blury.
    computation c_input("[N]->{c_input[i,j]: 0<=i<N and 0<=j<N}", expr(), false, p_uint8, &blurxy);

    expr e1 = (c_input(var("i") - 1, var("j")) +
               c_input(var("i")    , var("j")) +
               c_input(var("i") + 1, var("j")))/((uint8_t) 3);

    computation c_blurx("[N,M]->{c_blurx[i,j]: 0<i<N and 0<j<M}", e1, true, p_uint8, &blurxy);

    expr e2 = (c_blurx(var("i"), var("j") - 1) +
               c_blurx(var("i"), var("j")) +
               c_blurx(var("i"), var("j") + 1))/((uint8_t) 3);

    computation c_blury("[N,M]->{c_blury[i,j]: 1<i<N-1 and 1<j<M-1}", e2, true, p_uint8, &blurxy);

    // Create a memory buffer (2 dimensional).
    buffer b_blurx("b_blurx", 2, {tiramisu::expr(SIZE0),tiramisu::expr(SIZE1)}, p_uint8, NULL, a_temporary, &blurxy);

    // Map the computations to a buffer.
    c_input.set_access("{c_input[i,j]->b_input[i,j]}");
    c_blurx.set_access("{c_blurx[i,j]->b_blurx[i,j]}");
    c_blury.set_access("{c_blury[i,j]->b_blury[i,j]}");

    // Set the schedule of each computation.
    // The identity schedule means that the program order is not modified
    // (i.e. no optimization is applied).
    c_blurx.tile(0,1,2,2);
    c_blurx.tag_gpu_level(0,1);
    c_blury.set_schedule("{c_blury[i,j]->c_blury[0,0,i,0,j,0]}");
    c_blury.after(c_blurx, computation::root_dimension);

    // Set the arguments to blurxy
    blurxy.set_arguments({&b_input, &b_blury});
    // Generate code
    blurxy.gen_time_processor_domain();
    blurxy.gen_isl_ast();
    blurxy.gen_halide_stmt();
    blurxy.gen_halide_obj("build/generated_fct_tutorial_02.o");

    // Some debugging
    blurxy.dump_iteration_domain();
    blurxy.dump_halide_stmt();

    // Dump all the fields of the blurxy class.
    blurxy.dump(true);

    return 0;
}
