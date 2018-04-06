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

int main(int argc, char **argv)
{
    // Set default tiramisu options.
    global::set_default_tiramisu_options();
    global::set_loop_iterator_default_data_type(p_int32);

    tiramisu::function blurxy_tiramisu("blurxy_tiramisu");

    Halide::Buffer<uint8_t> in_image = Halide::Tools::load_image("./images/rgb.png");
    int SIZE0 = in_image.extent(0);
    int SIZE1 = in_image.extent(1);
    int SIZE2 = in_image.extent(2);

    // Output buffers.
    int blur_y_extent_2 = SIZE2;
    int blur_y_extent_1 = SIZE1 - 8;
    int blur_y_extent_0 = SIZE0 - 8;
    tiramisu::buffer buff_blur_y("buff_blur_y", {tiramisu::expr(blur_y_extent_2), tiramisu::expr(blur_y_extent_1), tiramisu::expr(blur_y_extent_0)}, tiramisu::p_uint8, tiramisu::a_output, &blurxy_tiramisu);

    // Input buffers.
    int p0_extent_2 = SIZE2;
    int p0_extent_1 = SIZE1;
    int p0_extent_0 = SIZE0;
    tiramisu::buffer buff_p0("buff_p0", {tiramisu::expr(p0_extent_2), tiramisu::expr(p0_extent_1), tiramisu::expr(p0_extent_0)}, tiramisu::p_uint8, tiramisu::a_input, &blurxy_tiramisu);
    tiramisu::computation p0("[p0_extent_2, p0_extent_1, p0_extent_0]->{p0[i2, i1, i0]: (0 <= i2 <= (p0_extent_2 + -1)) and (0 <= i1 <= (p0_extent_1 + -1)) and (0 <= i0 <= (p0_extent_0 + -1))}", expr(), false, tiramisu::p_uint8, &blurxy_tiramisu);
    p0.set_access("{p0[i2, i1, i0]->buff_p0[i2, i1, i0]}");


    // Define temporary buffers for "blur_x".
    tiramisu::buffer buff_blur_x("buff_blur_x", {tiramisu::expr(blur_y_extent_2), tiramisu::expr(blur_y_extent_1 + 2), tiramisu::expr(blur_y_extent_0)}, tiramisu::p_uint8, tiramisu::a_temporary, &blurxy_tiramisu);

    // Define loop bounds for dimension "blur_x_s0_c".
    tiramisu::constant blur_x_s0_c_loop_min("blur_x_s0_c_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &blurxy_tiramisu);
    tiramisu::constant blur_x_s0_c_loop_extent("blur_x_s0_c_loop_extent", tiramisu::expr(blur_y_extent_2), tiramisu::p_int32, true, NULL, 0, &blurxy_tiramisu);

    // Define loop bounds for dimension "blur_x_s0_y".
    tiramisu::constant blur_x_s0_y_loop_min("blur_x_s0_y_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &blurxy_tiramisu);
    tiramisu::constant blur_x_s0_y_loop_extent("blur_x_s0_y_loop_extent", (tiramisu::expr(blur_y_extent_1) + tiramisu::expr((int32_t)2)), tiramisu::p_int32, true, NULL, 0, &blurxy_tiramisu);

    // Define loop bounds for dimension "blur_x_s0_x".
    tiramisu::constant blur_x_s0_x_loop_min("blur_x_s0_x_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &blurxy_tiramisu);
    tiramisu::constant blur_x_s0_x_loop_extent("blur_x_s0_x_loop_extent", tiramisu::expr(blur_y_extent_0), tiramisu::p_int32, true, NULL, 0, &blurxy_tiramisu);
    tiramisu::computation blur_x_s0("[blur_x_s0_c_loop_min, blur_x_s0_c_loop_extent, blur_x_s0_y_loop_min, blur_x_s0_y_loop_extent, blur_x_s0_x_loop_min, blur_x_s0_x_loop_extent]->{blur_x_s0[blur_x_s0_c, blur_x_s0_y, blur_x_s0_x]: "
                        "(blur_x_s0_c_loop_min <= blur_x_s0_c <= ((blur_x_s0_c_loop_min + blur_x_s0_c_loop_extent) + -1)) and (blur_x_s0_y_loop_min <= blur_x_s0_y <= ((blur_x_s0_y_loop_min + blur_x_s0_y_loop_extent) + -1)) and (blur_x_s0_x_loop_min <= blur_x_s0_x <= ((blur_x_s0_x_loop_min + blur_x_s0_x_loop_extent) + -1))}",
                        (((p0(tiramisu::var("blur_x_s0_c"), tiramisu::var("blur_x_s0_y"), tiramisu::var("blur_x_s0_x")) + p0(tiramisu::var("blur_x_s0_c"), tiramisu::var("blur_x_s0_y"), (tiramisu::var("blur_x_s0_x") + tiramisu::expr((int32_t)1)))) + p0(tiramisu::var("blur_x_s0_c"), tiramisu::var("blur_x_s0_y"), (tiramisu::var("blur_x_s0_x") + tiramisu::expr((int32_t)2))))/tiramisu::expr((uint8_t)3)), true, tiramisu::p_uint8, &blurxy_tiramisu);
    blur_x_s0.set_access("{blur_x_s0[blur_x_s0_c, blur_x_s0_y, blur_x_s0_x]->buff_blur_x[blur_x_s0_c, blur_x_s0_y, blur_x_s0_x]}");

    // Define loop bounds for dimension "blur_y_s0_c".
    tiramisu::constant blur_y_s0_c_loop_min("blur_y_s0_c_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &blurxy_tiramisu);
    tiramisu::constant blur_y_s0_c_loop_extent("blur_y_s0_c_loop_extent", tiramisu::expr(blur_y_extent_2), tiramisu::p_int32, true, NULL, 0, &blurxy_tiramisu);

    // Define loop bounds for dimension "blur_y_s0_y".
    tiramisu::constant blur_y_s0_y_loop_min("blur_y_s0_y_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &blurxy_tiramisu);
    tiramisu::constant blur_y_s0_y_loop_extent("blur_y_s0_y_loop_extent", tiramisu::expr(blur_y_extent_1), tiramisu::p_int32, true, NULL, 0, &blurxy_tiramisu);

    // Define loop bounds for dimension "blur_y_s0_x".
    tiramisu::constant blur_y_s0_x_loop_min("blur_y_s0_x_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &blurxy_tiramisu);
    tiramisu::constant blur_y_s0_x_loop_extent("blur_y_s0_x_loop_extent", tiramisu::expr(blur_y_extent_0), tiramisu::p_int32, true, NULL, 0, &blurxy_tiramisu);
    tiramisu::computation blur_y_s0("[blur_y_s0_c_loop_min, blur_y_s0_c_loop_extent, blur_y_s0_y_loop_min, blur_y_s0_y_loop_extent, blur_y_s0_x_loop_min, blur_y_s0_x_loop_extent]->{blur_y_s0[blur_y_s0_c, blur_y_s0_y, blur_y_s0_x]: "
                        "(blur_y_s0_c_loop_min <= blur_y_s0_c <= ((blur_y_s0_c_loop_min + blur_y_s0_c_loop_extent) + -1)) and (blur_y_s0_y_loop_min <= blur_y_s0_y <= ((blur_y_s0_y_loop_min + blur_y_s0_y_loop_extent) + -1)) and (blur_y_s0_x_loop_min <= blur_y_s0_x <= ((blur_y_s0_x_loop_min + blur_y_s0_x_loop_extent) + -1))}",
                        (((blur_x_s0(tiramisu::var("blur_y_s0_c"), tiramisu::var("blur_y_s0_y"), tiramisu::var("blur_y_s0_x")) + blur_x_s0(tiramisu::var("blur_y_s0_c"), (tiramisu::var("blur_y_s0_y") + tiramisu::expr((int32_t)1)), tiramisu::var("blur_y_s0_x"))) + blur_x_s0(tiramisu::var("blur_y_s0_c"), (tiramisu::var("blur_y_s0_y") + tiramisu::expr((int32_t)2)), tiramisu::var("blur_y_s0_x")))/tiramisu::expr((uint8_t)3)), true, tiramisu::p_uint8, &blurxy_tiramisu);
    blur_y_s0.set_access("{blur_y_s0[blur_y_s0_c, blur_y_s0_y, blur_y_s0_x]->buff_blur_y[blur_y_s0_c, blur_y_s0_y, blur_y_s0_x]}");

    // Declare vars.
    tiramisu::var blur_y_s0_c("blur_y_s0_c");
    tiramisu::var blur_y_s0_x("blur_y_s0_x");
    tiramisu::var blur_y_s0_x_x_inner("blur_y_s0_x_x_inner");
    tiramisu::var blur_y_s0_x_x_outer("blur_y_s0_x_x_outer");
    tiramisu::var blur_y_s0_y("blur_y_s0_y");
    tiramisu::var blur_y_s0_y_y_inner("blur_y_s0_y_y_inner");
    tiramisu::var blur_y_s0_y_y_outer("blur_y_s0_y_y_outer");

    tiramisu::var blur_x_s0_c("blur_x_s0_c");
    tiramisu::var blur_x_s0_x("blur_x_s0_x");
    tiramisu::var blur_x_s0_x_outer("blur_x_s0_x_outer");
    tiramisu::var blur_x_s0_x_inner("blur_x_s0_x_inner");


    blurxy_tiramisu.add_context_constraints("[blur_y_s0_c_loop_min, blur_y_s0_c_loop_extent, blur_y_s0_y_loop_min, blur_y_s0_y_loop_extent, blur_y_s0_x_loop_min, blur_y_s0_x_loop_extent,blur_x_s0_c_loop_min, blur_x_s0_c_loop_extent, blur_x_s0_y_loop_min, blur_x_s0_y_loop_extent, blur_x_s0_x_loop_min, blur_x_s0_x_loop_extent]->{: blur_y_s0_c_loop_min=0 and blur_y_s0_y_loop_min=0 and blur_y_s0_x_loop_min=0 and blur_x_s0_c_loop_min=0 and blur_x_s0_y_loop_min=0 and blur_x_s0_x_loop_min=0 and blur_y_s0_y_loop_extent%8=0 and blur_x_s0_y_loop_extent%8=0 and blur_x_s0_x_loop_extent%8=0 and blur_y_s0_x_loop_extent%8=0 and blur_y_s0_c_loop_extent>0 and blur_y_s0_y_loop_extent>0 and blur_y_s0_x_loop_extent>0 and blur_x_s0_c_loop_extent>0 and blur_x_s0_y_loop_extent>0 and blur_x_s0_x_loop_extent>0}");

    // Add schedules.
    blur_y_s0.split(blur_y_s0_y, 8, blur_y_s0_y_y_outer, blur_y_s0_y_y_inner);
    blur_y_s0.parallelize(blur_y_s0_y_y_outer);
    blur_y_s0.parallelize(blur_y_s0_c);
    blur_y_s0.vectorize(blur_y_s0_x, 8);

    blur_x_s0.compute_at(blur_y_s0, blur_y_s0_y_y_inner);
    blur_x_s0.parallelize(blur_x_s0_c);
    blur_x_s0.split(blur_x_s0_x, 8, blur_x_s0_x_outer, blur_x_s0_x_inner);
    blur_x_s0.tag_vector_level(blur_x_s0_x_inner, 8);

    blurxy_tiramisu.set_arguments({&buff_p0, &buff_blur_y});
    blurxy_tiramisu.gen_time_space_domain();
    blurxy_tiramisu.gen_isl_ast();
    blurxy_tiramisu.gen_halide_stmt();
    blurxy_tiramisu.dump_halide_stmt();
    blurxy_tiramisu.gen_halide_obj("build/generated_fct_blurxy.o");

    return 0;
}
