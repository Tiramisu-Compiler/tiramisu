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

/* Halide code.
Func blurxy(Func input, Func blur_y) {
    Func blur_x, blur_y;
    Var x, y, xi, yi;

    blur_x(x, y) = (input(x-1, y) + input(x, y) + input(x+1, y))/3;
    blur_y(x, y) = (blur_x(x, y-1) + blur_x(x, y) + blur_x(x, y+1))/3;

    blur_y.split(y, y, yi, 8).parallel(y).vectorize(x, 8);
    blur_x.store_at(blur_y, y).compute_at(blur_y, yi).vectorize(x, 8);
  }
*/

using namespace coli;

int main(int argc, char **argv)
{
    // Set default coli options.
    global::set_default_coli_options();

    // A hack to represent the image size.  TODO: this should be retrieved from
    // the function arguments.
    Halide::Image<uint16_t> in_image = Halide::Tools::load_image("./images/rgb.png");
    int  SIZE0 = in_image.extent(0);
    int  SIZE1 = in_image.extent(1);

    /*
     * Declare a function blurxy.
     * Declare two arguments (coli buffers) for the function: b_input and b_blury
     * Declare an invariant for the function.
     */
    function blurxy("blurxy_coli");
    buffer b_input("b_input", 2, {coli::expr(SIZE0),coli::expr(SIZE1)}, p_uint16, NULL, a_input, &blurxy);
    buffer b_blury("b_blury", 2, {coli::expr(SIZE0),coli::expr(SIZE1)}, p_uint16, NULL, a_output, &blurxy);
    expr e_p0 = expr((int32_t) SIZE0);
    expr e_p1 = expr((int32_t) SIZE1);
    constant p0("N", &e_p0, p_int32, true, NULL, 0, &blurxy);
    constant p1("M", &e_p1, p_int32, true, NULL, 0, &blurxy);

    // Declare the computations c_blurx and c_blury.
    computation c_input("[N]->{c_input[i,j]: 0<=i<N and 0<=j<N}", NULL, false, p_uint16, &blurxy);

    idx i = idx("i");
    idx j = idx("j");

    expr e1 = (c_input(i-1, j) + c_input(i, j) + c_input(i+1, j))/((uint16_t) 3);
    computation c_blurx("[N,M]->{c_blurx[i,j]: 0<i<N-1 and 0<j<M-1}", &e1, true, p_uint16, &blurxy);

    expr e2 = (c_blurx(i, j-1) + c_blurx(i, j) + c_blurx(i, j+1))/((uint16_t) 3);
    computation c_blury("[N,M]->{c_blury[i,j]: 1<i<N-2 and 1<j<M-2}", &e2, true, p_uint16, &blurxy);

    // Create a memory buffer (2 dimensional).
    buffer b_blurx("b_blurx", 2, {coli::expr(SIZE0),coli::expr(SIZE1)}, p_uint16, NULL, a_temporary, &blurxy);

    // Map the computations to a buffer.
    c_input.set_access("{c_input[i,j]->b_input[i,j]}");
    c_blurx.set_access("{c_blurx[i,j]->b_blurx[i,j]}");
    c_blury.set_access("{c_blury[i,j]->b_blury[i,j]}");

    // Set the schedule of each computation.
    //c_blury.split(0, 8);
    //c_blury.tag_parallel_dimension(0);
    //c_blury.split(2, 8);
    //c_blury.tag_vector_dimension(3);
    c_blury.after(c_blurx, computation::root_dimension);

    // Set the arguments to blurxy
    blurxy.set_arguments({&b_input, &b_blury});
    // Generate code
    blurxy.gen_isl_ast();
    blurxy.gen_halide_stmt();
    blurxy.gen_halide_obj("build/generated_fct_blurxy.o");

    // Some debugging
    blurxy.dump_iteration_domain();
    blurxy.dump_halide_stmt();

    // Dump all the fields of the blurxy class.
    blurxy.dump(true);

    return 0;
}
