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

    tiramisu::function fusion_tiramisu("fusion_tiramisu");

    Halide::Image<uint8_t> in_image = Halide::Tools::load_image("./images/rgb.png");
    int SIZE0 = in_image.extent(0);
    int SIZE1 = in_image.extent(1);
    int SIZE2 = in_image.extent(2);

    // Output buffers.
    int f_extent_2 = SIZE2;
    int f_extent_1 = SIZE1;
    int f_extent_0 = SIZE0;
    tiramisu::buffer buff_f("buff_f", 3, {tiramisu::expr(f_extent_2), tiramisu::expr(f_extent_1), tiramisu::expr(f_extent_0)}, tiramisu::p_uint8, NULL, tiramisu::a_output, &fusion_tiramisu);
    tiramisu::buffer buff_g("buff_g", 3, {tiramisu::expr(f_extent_2), tiramisu::expr(f_extent_1), tiramisu::expr(f_extent_0)}, tiramisu::p_uint8, NULL, tiramisu::a_output, &fusion_tiramisu);
    tiramisu::buffer buff_h("buff_h", 3, {tiramisu::expr(f_extent_2), tiramisu::expr(f_extent_1), tiramisu::expr(f_extent_0)}, tiramisu::p_uint8, NULL, tiramisu::a_output, &fusion_tiramisu);
    tiramisu::buffer buff_k("buff_k", 3, {tiramisu::expr(f_extent_2), tiramisu::expr(f_extent_1), tiramisu::expr(f_extent_0)}, tiramisu::p_uint8, NULL, tiramisu::a_output, &fusion_tiramisu);

    // Input buffers.
    int input_extent_2 = SIZE2;
    int input_extent_1 = SIZE1;
    int input_extent_0 = SIZE0;
    tiramisu::buffer buff_input("buff_input", 3, {tiramisu::expr(input_extent_2), tiramisu::expr(input_extent_1), tiramisu::expr(input_extent_0)}, tiramisu::p_uint8, NULL, tiramisu::a_input, &fusion_tiramisu);
    tiramisu::computation input("[input_extent_2, input_extent_1, input_extent_0]->{input[i2, i1, i0]: (0 <= i2 <= (input_extent_2 + -1)) and (0 <= i1 <= (input_extent_1 + -1)) and (0 <= i0 <= (input_extent_0 + -1))}", expr(), false, tiramisu::p_uint8, &fusion_tiramisu);
    input.set_access("{input[i2, i1, i0]->buff_input[i2, i1, i0]}");



    tiramisu::constant f_s0_c_loop_min("f_s0_c_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &fusion_tiramisu);
    tiramisu::constant f_s0_c_loop_extent("f_s0_c_loop_extent", tiramisu::expr(f_extent_2), tiramisu::p_int32, true, NULL, 0, &fusion_tiramisu);
    tiramisu::constant f_s0_y_loop_min("f_s0_y_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &fusion_tiramisu);
    tiramisu::constant f_s0_y_loop_extent("f_s0_y_loop_extent", tiramisu::expr(f_extent_1), tiramisu::p_int32, true, NULL, 0, &fusion_tiramisu);
    tiramisu::constant f_s0_x_loop_min("f_s0_x_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &fusion_tiramisu);
    tiramisu::constant f_s0_x_loop_extent("f_s0_x_loop_extent", tiramisu::expr(f_extent_0), tiramisu::p_int32, true, NULL, 0, &fusion_tiramisu);
    tiramisu::computation f_s0("[f_s0_c_loop_min, f_s0_c_loop_extent, f_s0_y_loop_min, f_s0_y_loop_extent, f_s0_x_loop_min, f_s0_x_loop_extent]->{f_s0[f_s0_c, f_s0_y, f_s0_x]: "
                        "(f_s0_c_loop_min <= f_s0_c <= ((f_s0_c_loop_min + f_s0_c_loop_extent) + -1)) and (f_s0_y_loop_min <= f_s0_y <= ((f_s0_y_loop_min + f_s0_y_loop_extent) + -1)) and (f_s0_x_loop_min <= f_s0_x <= ((f_s0_x_loop_min + f_s0_x_loop_extent) + -1))}",
                        (tiramisu::expr((uint8_t)255) - input(tiramisu::idx("f_s0_c"), tiramisu::idx("f_s0_y"), tiramisu::idx("f_s0_x"))), true, tiramisu::p_uint8, &fusion_tiramisu);
    f_s0.set_access("{f_s0[f_s0_c, f_s0_y, f_s0_x]->buff_f[f_s0_c, f_s0_y, f_s0_x]}");



    // Define loop bounds for dimension "g_s0_c".
    tiramisu::constant g_s0_c_loop_min("g_s0_c_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &fusion_tiramisu);
    tiramisu::constant g_s0_c_loop_extent("g_s0_c_loop_extent", tiramisu::expr(f_extent_2), tiramisu::p_int32, true, NULL, 0, &fusion_tiramisu);
    tiramisu::constant g_s0_y_loop_min("g_s0_y_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &fusion_tiramisu);
    tiramisu::constant g_s0_y_loop_extent("g_s0_y_loop_extent", tiramisu::expr(f_extent_1), tiramisu::p_int32, true, NULL, 0, &fusion_tiramisu);
    tiramisu::constant g_s0_x_loop_min("g_s0_x_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &fusion_tiramisu);
    tiramisu::constant g_s0_x_loop_extent("g_s0_x_loop_extent", tiramisu::expr(f_extent_0), tiramisu::p_int32, true, NULL, 0, &fusion_tiramisu);
    tiramisu::computation g_s0("[g_s0_c_loop_min, g_s0_c_loop_extent, g_s0_y_loop_min, g_s0_y_loop_extent, g_s0_x_loop_min, g_s0_x_loop_extent]->{g_s0[g_s0_c, g_s0_y, g_s0_x]: "
                        "(g_s0_c_loop_min <= g_s0_c <= ((g_s0_c_loop_min + g_s0_c_loop_extent) + -1)) and (g_s0_y_loop_min <= g_s0_y <= ((g_s0_y_loop_min + g_s0_y_loop_extent) + -1)) and (g_s0_x_loop_min <= g_s0_x <= ((g_s0_x_loop_min + g_s0_x_loop_extent) + -1))}",
                        (tiramisu::expr((uint8_t)2) * input(tiramisu::idx("g_s0_c"), tiramisu::idx("g_s0_y"), tiramisu::idx("g_s0_x"))), true, tiramisu::p_uint8, &fusion_tiramisu);
    g_s0.set_access("{g_s0[g_s0_c, g_s0_y, g_s0_x]->buff_g[g_s0_c, g_s0_y, g_s0_x]}");



    // Loop over h
    tiramisu::constant h_s0_c_loop_min("h_s0_c_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &fusion_tiramisu);
    tiramisu::constant h_s0_c_loop_extent("h_s0_c_loop_extent", tiramisu::expr(f_extent_2), tiramisu::p_int32, true, NULL, 0, &fusion_tiramisu);
    tiramisu::constant h_s0_y_loop_min("h_s0_y_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &fusion_tiramisu);
    tiramisu::constant h_s0_y_loop_extent("h_s0_y_loop_extent", tiramisu::expr(f_extent_1), tiramisu::p_int32, true, NULL, 0, &fusion_tiramisu);
    tiramisu::constant h_s0_x_loop_min("h_s0_x_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &fusion_tiramisu);
    tiramisu::constant h_s0_x_loop_extent("h_s0_x_loop_extent", tiramisu::expr(f_extent_0), tiramisu::p_int32, true, NULL, 0, &fusion_tiramisu);
    tiramisu::computation h_s0("[h_s0_c_loop_min, h_s0_c_loop_extent, h_s0_y_loop_min, h_s0_y_loop_extent, h_s0_x_loop_min, h_s0_x_loop_extent]->{h_s0[h_s0_c, h_s0_y, h_s0_x]: "
                        "(h_s0_c_loop_min <= h_s0_c <= ((h_s0_c_loop_min + h_s0_c_loop_extent) + -1)) and (h_s0_y_loop_min <= h_s0_y <= ((h_s0_y_loop_min + h_s0_y_loop_extent) + -1)) and (h_s0_x_loop_min <= h_s0_x <= ((h_s0_x_loop_min + h_s0_x_loop_extent) + -1))}",
                        (tiramisu::expr((uint8_t)2) * input(tiramisu::idx("h_s0_c"), tiramisu::idx("h_s0_y"), tiramisu::idx("h_s0_x"))), true, tiramisu::p_uint8, &fusion_tiramisu);
    h_s0.set_access("{h_s0[h_s0_c, h_s0_y, h_s0_x]->buff_h[h_s0_c, h_s0_y, h_s0_x]}");



    tiramisu::constant k_s0_c_loop_min("k_s0_c_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &fusion_tiramisu);
    tiramisu::constant k_s0_c_loop_extent("k_s0_c_loop_extent", tiramisu::expr(f_extent_2), tiramisu::p_int32, true, NULL, 0, &fusion_tiramisu);
    tiramisu::constant k_s0_y_loop_min("k_s0_y_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &fusion_tiramisu);
    tiramisu::constant k_s0_y_loop_extent("k_s0_y_loop_extent", tiramisu::expr(f_extent_1), tiramisu::p_int32, true, NULL, 0, &fusion_tiramisu);
    tiramisu::constant k_s0_x_loop_min("k_s0_x_loop_min", tiramisu::expr((int32_t)0), tiramisu::p_int32, true, NULL, 0, &fusion_tiramisu);
    tiramisu::constant k_s0_x_loop_extent("k_s0_x_loop_extent", tiramisu::expr(f_extent_0), tiramisu::p_int32, true, NULL, 0, &fusion_tiramisu);
    tiramisu::computation k_s0("[k_s0_c_loop_min, k_s0_c_loop_extent, k_s0_y_loop_min, k_s0_y_loop_extent, k_s0_x_loop_min, k_s0_x_loop_extent]->{k_s0[k_s0_c, k_s0_y, k_s0_x]: "
                         "(k_s0_c_loop_min <= k_s0_c <= ((k_s0_c_loop_min + k_s0_c_loop_extent) + -1)) and (k_s0_y_loop_min <= k_s0_y <= ((k_s0_y_loop_min + k_s0_y_loop_extent) + -1)) and (k_s0_x_loop_min <= k_s0_x <= ((k_s0_x_loop_min + k_s0_x_loop_extent) + -1))}",
                         (tiramisu::expr((uint8_t)2) * input(tiramisu::idx("k_s0_c"), tiramisu::idx("k_s0_y"), tiramisu::idx("k_s0_x"))), true, tiramisu::p_uint8, &fusion_tiramisu);
    k_s0.set_access("{k_s0[k_s0_c, k_s0_y, k_s0_x]->buff_k[k_s0_c, k_s0_y, k_s0_x]}");



    fusion_tiramisu.set_context_set("[f_s0_c_loop_min, f_s0_c_loop_extent, f_s0_y_loop_min, f_s0_y_loop_extent, f_s0_x_loop_min, f_s0_x_loop_extent, g_s0_c_loop_min, g_s0_c_loop_extent, g_s0_y_loop_min, g_s0_y_loop_extent, g_s0_x_loop_min, g_s0_x_loop_extent, h_s0_c_loop_min, h_s0_c_loop_extent, h_s0_y_loop_min, h_s0_y_loop_extent, h_s0_x_loop_min, h_s0_x_loop_extent, k_s0_c_loop_min, k_s0_c_loop_extent, k_s0_y_loop_min, k_s0_y_loop_extent, k_s0_x_loop_min, k_s0_x_loop_extent]->{: k_s0_x_loop_extent=f_s0_x_loop_extent and k_s0_x_loop_min=f_s0_x_loop_min and k_s0_y_loop_extent=f_s0_y_loop_extent and k_s0_y_loop_min=f_s0_y_loop_min and k_s0_c_loop_extent=f_s0_c_loop_extent and k_s0_c_loop_min=f_s0_c_loop_min and         h_s0_x_loop_extent=f_s0_x_loop_extent and h_s0_x_loop_min=f_s0_x_loop_min and h_s0_y_loop_extent=f_s0_y_loop_extent and h_s0_y_loop_min=f_s0_y_loop_min and h_s0_c_loop_extent=f_s0_c_loop_extent and f_s0_c_loop_min=h_s0_c_loop_min and f_s0_c_loop_min=g_s0_c_loop_min and f_s0_c_loop_extent=g_s0_c_loop_extent and f_s0_y_loop_min=g_s0_y_loop_min and f_s0_y_loop_extent=g_s0_y_loop_extent and f_s0_c_loop_min>=0 and f_s0_c_loop_extent>0}");

    f_s0.set_schedule("[f_s0_c_loop_min, f_s0_c_loop_extent, f_s0_y_loop_min, f_s0_y_loop_extent, f_s0_x_loop_min, f_s0_x_loop_extent,    g_s0_c_loop_min, g_s0_c_loop_extent, g_s0_y_loop_min, g_s0_y_loop_extent, g_s0_x_loop_min, g_s0_x_loop_extent,    h_s0_c_loop_min, h_s0_c_loop_extent, h_s0_y_loop_min, h_s0_y_loop_extent, h_s0_x_loop_min, h_s0_x_loop_extent,   k_s0_c_loop_min, k_s0_c_loop_extent, k_s0_y_loop_min, k_s0_y_loop_extent, k_s0_x_loop_min, k_s0_x_loop_extent]->{f_s0[f_s0_c, f_s0_y, f_s0_x] -> f_s0[0, f_s0_c, 0, f_s0_y, 0, f_s0_x, 0]}");
    g_s0.set_schedule("[f_s0_c_loop_min, f_s0_c_loop_extent, f_s0_y_loop_min, f_s0_y_loop_extent, f_s0_x_loop_min, f_s0_x_loop_extent,    g_s0_c_loop_min, g_s0_c_loop_extent, g_s0_y_loop_min, g_s0_y_loop_extent, g_s0_x_loop_min, g_s0_x_loop_extent,    h_s0_c_loop_min, h_s0_c_loop_extent, h_s0_y_loop_min, h_s0_y_loop_extent, h_s0_x_loop_min, h_s0_x_loop_extent,   k_s0_c_loop_min, k_s0_c_loop_extent, k_s0_y_loop_min, k_s0_y_loop_extent, k_s0_x_loop_min, k_s0_x_loop_extent]->{g_s0[g_s0_c, g_s0_y, g_s0_x] -> g_s0[0, g_s0_c, 0, g_s0_y, 0, g_s0_x, 1]}");
    h_s0.set_schedule("[f_s0_c_loop_min, f_s0_c_loop_extent, f_s0_y_loop_min, f_s0_y_loop_extent, f_s0_x_loop_min, f_s0_x_loop_extent,    g_s0_c_loop_min, g_s0_c_loop_extent, g_s0_y_loop_min, g_s0_y_loop_extent, g_s0_x_loop_min, g_s0_x_loop_extent,    h_s0_c_loop_min, h_s0_c_loop_extent, h_s0_y_loop_min, h_s0_y_loop_extent, h_s0_x_loop_min, h_s0_x_loop_extent,   k_s0_c_loop_min, k_s0_c_loop_extent, k_s0_y_loop_min, k_s0_y_loop_extent, k_s0_x_loop_min, k_s0_x_loop_extent]->{h_s0[h_s0_c, h_s0_y, h_s0_x] -> h_s0[0, h_s0_c, 0, h_s0_y, 0, h_s0_x, 2]}");
    k_s0.set_schedule("[f_s0_c_loop_min, f_s0_c_loop_extent, f_s0_y_loop_min, f_s0_y_loop_extent, f_s0_x_loop_min, f_s0_x_loop_extent,    g_s0_c_loop_min, g_s0_c_loop_extent, g_s0_y_loop_min, g_s0_y_loop_extent, g_s0_x_loop_min, g_s0_x_loop_extent,    h_s0_c_loop_min, h_s0_c_loop_extent, h_s0_y_loop_min, h_s0_y_loop_extent, h_s0_x_loop_min, h_s0_x_loop_extent,   k_s0_c_loop_min, k_s0_c_loop_extent, k_s0_y_loop_min, k_s0_y_loop_extent, k_s0_x_loop_min, k_s0_x_loop_extent]->{k_s0[k_s0_c, k_s0_y, k_s0_x] -> k_s0[0, k_s0_c, 0, k_s0_y, 0, k_s0_x, 3]}");
    f_s0.tag_parallel_dimension(0);
    f_s0.tag_parallel_dimension(1);
    g_s0.tag_parallel_dimension(0);
    g_s0.tag_parallel_dimension(1);
    h_s0.tag_parallel_dimension(0);
    h_s0.tag_parallel_dimension(1);
    k_s0.tag_parallel_dimension(0);
    k_s0.tag_parallel_dimension(1);
    f_s0.tag_vector_dimension(2);
    g_s0.tag_vector_dimension(2);
    f_s0.tag_vector_dimension(3);
    g_s0.tag_vector_dimension(3);


    fusion_tiramisu.set_arguments({&buff_input, &buff_f, &buff_g, &buff_h, &buff_k});
    fusion_tiramisu.gen_time_processor_domain();
    fusion_tiramisu.gen_isl_ast();
    fusion_tiramisu.gen_halide_stmt();
    fusion_tiramisu.dump_halide_stmt();
    fusion_tiramisu.gen_halide_obj("build/generated_fct_fusion.o");

    return 0;
}

