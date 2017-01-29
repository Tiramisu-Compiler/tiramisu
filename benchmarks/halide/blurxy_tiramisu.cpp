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


using namespace coli;

int main(int argc, char **argv)
{
    // Set default coli options.
    global::set_default_coli_options();

    coli::function blurxy_coli("blurxy_coli");

    Halide::Image<uint8_t> in_image = Halide::Tools::load_image("./images/rgb.png");
    int SIZE0 = in_image.extent(0);
    int SIZE1 = in_image.extent(1);
    int SIZE2 = in_image.extent(2);

    // Output buffers.
    int blur_y_extent_2 = SIZE2;
    int blur_y_extent_1 = SIZE1 - 8;
    int blur_y_extent_0 = SIZE0 - 8;
    coli::buffer buff_blur_y("buff_blur_y", 3, {coli::expr(blur_y_extent_2), coli::expr(blur_y_extent_1), coli::expr(blur_y_extent_0)}, coli::p_uint8, NULL, coli::a_output, &blurxy_coli);

    // Input buffers.
    int p0_extent_2 = SIZE2;
    int p0_extent_1 = SIZE1;
    int p0_extent_0 = SIZE0;
    coli::buffer buff_p0("buff_p0", 3, {coli::expr(p0_extent_2), coli::expr(p0_extent_1), coli::expr(p0_extent_0)}, coli::p_uint8, NULL, coli::a_input, &blurxy_coli);
    coli::computation p0("[p0_extent_2, p0_extent_1, p0_extent_0]->{p0[i2, i1, i0]: (0 <= i2 <= (p0_extent_2 + -1)) and (0 <= i1 <= (p0_extent_1 + -1)) and (0 <= i0 <= (p0_extent_0 + -1))}", expr(), false, coli::p_uint8, &blurxy_coli);
    p0.set_access("{p0[i2, i1, i0]->buff_p0[i2, i1, i0]}");


    // Define temporary buffers for "blur_x".
    coli::buffer buff_blur_x("buff_blur_x", 3, {coli::expr(blur_y_extent_2), coli::expr(blur_y_extent_1 + 2), coli::expr(blur_y_extent_0)}, coli::p_uint8, NULL, coli::a_temporary, &blurxy_coli);

    // Define loop bounds for dimension "blur_x_s0_c".
    coli::constant blur_x_s0_c_loop_min("blur_x_s0_c_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &blurxy_coli);
    coli::constant blur_x_s0_c_loop_extent("blur_x_s0_c_loop_extent", coli::expr(blur_y_extent_2), coli::p_int32, true, NULL, 0, &blurxy_coli);

    // Define loop bounds for dimension "blur_x_s0_y".
    coli::constant blur_x_s0_y_loop_min("blur_x_s0_y_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &blurxy_coli);
    coli::constant blur_x_s0_y_loop_extent("blur_x_s0_y_loop_extent", (coli::expr(blur_y_extent_1) + coli::expr((int32_t)2)), coli::p_int32, true, NULL, 0, &blurxy_coli);

    // Define loop bounds for dimension "blur_x_s0_x".
    coli::constant blur_x_s0_x_loop_min("blur_x_s0_x_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &blurxy_coli);
    coli::constant blur_x_s0_x_loop_extent("blur_x_s0_x_loop_extent", coli::expr(blur_y_extent_0), coli::p_int32, true, NULL, 0, &blurxy_coli);
    coli::computation blur_x_s0("[blur_x_s0_c_loop_min, blur_x_s0_c_loop_extent, blur_x_s0_y_loop_min, blur_x_s0_y_loop_extent, blur_x_s0_x_loop_min, blur_x_s0_x_loop_extent]->{blur_x_s0[blur_x_s0_c, blur_x_s0_y, blur_x_s0_x]: "
                        "(blur_x_s0_c_loop_min <= blur_x_s0_c <= ((blur_x_s0_c_loop_min + blur_x_s0_c_loop_extent) + -1)) and (blur_x_s0_y_loop_min <= blur_x_s0_y <= ((blur_x_s0_y_loop_min + blur_x_s0_y_loop_extent) + -1)) and (blur_x_s0_x_loop_min <= blur_x_s0_x <= ((blur_x_s0_x_loop_min + blur_x_s0_x_loop_extent) + -1))}",
                        (((p0(coli::idx("blur_x_s0_c"), coli::idx("blur_x_s0_y"), coli::idx("blur_x_s0_x")) + p0(coli::idx("blur_x_s0_c"), coli::idx("blur_x_s0_y"), (coli::idx("blur_x_s0_x") + coli::expr((int32_t)1)))) + p0(coli::idx("blur_x_s0_c"), coli::idx("blur_x_s0_y"), (coli::idx("blur_x_s0_x") + coli::expr((int32_t)2))))/coli::expr((uint8_t)3)), true, coli::p_uint8, &blurxy_coli);
    blur_x_s0.set_access("{blur_x_s0[blur_x_s0_c, blur_x_s0_y, blur_x_s0_x]->buff_blur_x[blur_x_s0_c, blur_x_s0_y, blur_x_s0_x]}");

    // Define compute level for "blur_x".
    blur_x_s0.first(computation::root_dimension);

    // Define loop bounds for dimension "blur_y_s0_c".
    coli::constant blur_y_s0_c_loop_min("blur_y_s0_c_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &blurxy_coli);
    coli::constant blur_y_s0_c_loop_extent("blur_y_s0_c_loop_extent", coli::expr(blur_y_extent_2), coli::p_int32, true, NULL, 0, &blurxy_coli);

    // Define loop bounds for dimension "blur_y_s0_y".
    coli::constant blur_y_s0_y_loop_min("blur_y_s0_y_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &blurxy_coli);
    coli::constant blur_y_s0_y_loop_extent("blur_y_s0_y_loop_extent", coli::expr(blur_y_extent_1), coli::p_int32, true, NULL, 0, &blurxy_coli);

    // Define loop bounds for dimension "blur_y_s0_x".
    coli::constant blur_y_s0_x_loop_min("blur_y_s0_x_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &blurxy_coli);
    coli::constant blur_y_s0_x_loop_extent("blur_y_s0_x_loop_extent", coli::expr(blur_y_extent_0), coli::p_int32, true, NULL, 0, &blurxy_coli);
    coli::computation blur_y_s0("[blur_y_s0_c_loop_min, blur_y_s0_c_loop_extent, blur_y_s0_y_loop_min, blur_y_s0_y_loop_extent, blur_y_s0_x_loop_min, blur_y_s0_x_loop_extent]->{blur_y_s0[blur_y_s0_c, blur_y_s0_y, blur_y_s0_x]: "
                        "(blur_y_s0_c_loop_min <= blur_y_s0_c <= ((blur_y_s0_c_loop_min + blur_y_s0_c_loop_extent) + -1)) and (blur_y_s0_y_loop_min <= blur_y_s0_y <= ((blur_y_s0_y_loop_min + blur_y_s0_y_loop_extent) + -1)) and (blur_y_s0_x_loop_min <= blur_y_s0_x <= ((blur_y_s0_x_loop_min + blur_y_s0_x_loop_extent) + -1))}",
                        (((blur_x_s0(coli::idx("blur_y_s0_c"), coli::idx("blur_y_s0_y"), coli::idx("blur_y_s0_x")) + blur_x_s0(coli::idx("blur_y_s0_c"), (coli::idx("blur_y_s0_y") + coli::expr((int32_t)1)), coli::idx("blur_y_s0_x"))) + blur_x_s0(coli::idx("blur_y_s0_c"), (coli::idx("blur_y_s0_y") + coli::expr((int32_t)2)), coli::idx("blur_y_s0_x")))/coli::expr((uint8_t)3)), true, coli::p_uint8, &blurxy_coli);
    blur_y_s0.set_access("{blur_y_s0[blur_y_s0_c, blur_y_s0_y, blur_y_s0_x]->buff_blur_y[blur_y_s0_c, blur_y_s0_y, blur_y_s0_x]}");

    // Define compute level for "blur_y".
    blur_y_s0.after(blur_x_s0, computation::root_dimension);

    // Add schedules.
    blur_x_s0.tag_parallel_dimension(1);
    blur_x_s0.tag_parallel_dimension(0);
    blur_y_s0.tag_parallel_dimension(1);
    blur_y_s0.tag_parallel_dimension(0);

    blurxy_coli.set_arguments({&buff_p0, &buff_blur_y});
    blurxy_coli.gen_time_processor_domain();
    blurxy_coli.gen_isl_ast();
    blurxy_coli.gen_halide_stmt();
    blurxy_coli.dump_halide_stmt();
    blurxy_coli.gen_halide_obj("build/generated_fct_blurxy.o");

    return 0;
}
