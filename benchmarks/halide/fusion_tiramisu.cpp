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

    coli::function fusion_coli("fusion_coli");

    Halide::Image<uint8_t> in_image = Halide::Tools::load_image("./images/rgb.png");
    int SIZE0 = in_image.extent(0);
    int SIZE1 = in_image.extent(1);
    int SIZE2 = in_image.extent(2);

    // Output buffers.
    int f_extent_2 = SIZE2;
    int f_extent_1 = SIZE1;
    int f_extent_0 = SIZE0;
    coli::buffer buff_f("buff_f", 3, {coli::expr(f_extent_2), coli::expr(f_extent_1), coli::expr(f_extent_0)}, coli::p_uint8, NULL, coli::a_output, &fusion_coli);
    coli::buffer buff_g("buff_g", 3, {coli::expr(f_extent_2), coli::expr(f_extent_1), coli::expr(f_extent_0)}, coli::p_uint8, NULL, coli::a_output, &fusion_coli);
    coli::buffer buff_h("buff_h", 3, {coli::expr(f_extent_2), coli::expr(f_extent_1), coli::expr(f_extent_0)}, coli::p_uint8, NULL, coli::a_output, &fusion_coli);
    coli::buffer buff_k("buff_k", 3, {coli::expr(f_extent_2), coli::expr(f_extent_1), coli::expr(f_extent_0)}, coli::p_uint8, NULL, coli::a_output, &fusion_coli);

    // Input buffers.
    int input_extent_2 = SIZE2;
    int input_extent_1 = SIZE1;
    int input_extent_0 = SIZE0;
    coli::buffer buff_input("buff_input", 3, {coli::expr(input_extent_2), coli::expr(input_extent_1), coli::expr(input_extent_0)}, coli::p_uint8, NULL, coli::a_input, &fusion_coli);
    coli::computation input("[input_extent_2, input_extent_1, input_extent_0]->{input[i2, i1, i0]: (0 <= i2 <= (input_extent_2 + -1)) and (0 <= i1 <= (input_extent_1 + -1)) and (0 <= i0 <= (input_extent_0 + -1))}", expr(), false, coli::p_uint8, &fusion_coli);
    input.set_access("{input[i2, i1, i0]->buff_input[i2, i1, i0]}");



    coli::constant f_s0_c_loop_min("f_s0_c_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &fusion_coli);
    coli::constant f_s0_c_loop_extent("f_s0_c_loop_extent", coli::expr(f_extent_2), coli::p_int32, true, NULL, 0, &fusion_coli);
    coli::constant f_s0_y_loop_min("f_s0_y_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &fusion_coli);
    coli::constant f_s0_y_loop_extent("f_s0_y_loop_extent", coli::expr(f_extent_1), coli::p_int32, true, NULL, 0, &fusion_coli);
    coli::constant f_s0_x_loop_min("f_s0_x_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &fusion_coli);
    coli::constant f_s0_x_loop_extent("f_s0_x_loop_extent", coli::expr(f_extent_0), coli::p_int32, true, NULL, 0, &fusion_coli);
    coli::computation f_s0("[f_s0_c_loop_min, f_s0_c_loop_extent, f_s0_y_loop_min, f_s0_y_loop_extent, f_s0_x_loop_min, f_s0_x_loop_extent]->{f_s0[f_s0_c, f_s0_y, f_s0_x]: "
                        "(f_s0_c_loop_min <= f_s0_c <= ((f_s0_c_loop_min + f_s0_c_loop_extent) + -1)) and (f_s0_y_loop_min <= f_s0_y <= ((f_s0_y_loop_min + f_s0_y_loop_extent) + -1)) and (f_s0_x_loop_min <= f_s0_x <= ((f_s0_x_loop_min + f_s0_x_loop_extent) + -1))}",
                        (coli::expr((uint8_t)255) - input(coli::idx("f_s0_c"), coli::idx("f_s0_y"), coli::idx("f_s0_x"))), true, coli::p_uint8, &fusion_coli);
    f_s0.set_access("{f_s0[f_s0_c, f_s0_y, f_s0_x]->buff_f[f_s0_c, f_s0_y, f_s0_x]}");



    // Define loop bounds for dimension "g_s0_c".
    coli::constant g_s0_c_loop_min("g_s0_c_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &fusion_coli);
    coli::constant g_s0_c_loop_extent("g_s0_c_loop_extent", coli::expr(f_extent_2), coli::p_int32, true, NULL, 0, &fusion_coli);
    coli::constant g_s0_y_loop_min("g_s0_y_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &fusion_coli);
    coli::constant g_s0_y_loop_extent("g_s0_y_loop_extent", coli::expr(f_extent_1), coli::p_int32, true, NULL, 0, &fusion_coli);
    coli::constant g_s0_x_loop_min("g_s0_x_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &fusion_coli);
    coli::constant g_s0_x_loop_extent("g_s0_x_loop_extent", coli::expr(f_extent_0), coli::p_int32, true, NULL, 0, &fusion_coli);
    coli::computation g_s0("[g_s0_c_loop_min, g_s0_c_loop_extent, g_s0_y_loop_min, g_s0_y_loop_extent, g_s0_x_loop_min, g_s0_x_loop_extent]->{g_s0[g_s0_c, g_s0_y, g_s0_x]: "
                        "(g_s0_c_loop_min <= g_s0_c <= ((g_s0_c_loop_min + g_s0_c_loop_extent) + -1)) and (g_s0_y_loop_min <= g_s0_y <= ((g_s0_y_loop_min + g_s0_y_loop_extent) + -1)) and (g_s0_x_loop_min <= g_s0_x <= ((g_s0_x_loop_min + g_s0_x_loop_extent) + -1))}",
                        (coli::expr((uint8_t)2) * input(coli::idx("g_s0_c"), coli::idx("g_s0_y"), coli::idx("g_s0_x"))), true, coli::p_uint8, &fusion_coli);
    g_s0.set_access("{g_s0[g_s0_c, g_s0_y, g_s0_x]->buff_g[g_s0_c, g_s0_y, g_s0_x]}");



    // Loop over h
    coli::constant h_s0_c_loop_min("h_s0_c_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &fusion_coli);
    coli::constant h_s0_c_loop_extent("h_s0_c_loop_extent", coli::expr(f_extent_2), coli::p_int32, true, NULL, 0, &fusion_coli);
    coli::constant h_s0_y_loop_min("h_s0_y_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &fusion_coli);
    coli::constant h_s0_y_loop_extent("h_s0_y_loop_extent", coli::expr(f_extent_1), coli::p_int32, true, NULL, 0, &fusion_coli);
    coli::constant h_s0_x_loop_min("h_s0_x_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &fusion_coli);
    coli::constant h_s0_x_loop_extent("h_s0_x_loop_extent", coli::expr(f_extent_0), coli::p_int32, true, NULL, 0, &fusion_coli);
    coli::computation h_s0("[h_s0_c_loop_min, h_s0_c_loop_extent, h_s0_y_loop_min, h_s0_y_loop_extent, h_s0_x_loop_min, h_s0_x_loop_extent]->{h_s0[h_s0_c, h_s0_y, h_s0_x]: "
                        "(h_s0_c_loop_min <= h_s0_c <= ((h_s0_c_loop_min + h_s0_c_loop_extent) + -1)) and (h_s0_y_loop_min <= h_s0_y <= ((h_s0_y_loop_min + h_s0_y_loop_extent) + -1)) and (h_s0_x_loop_min <= h_s0_x <= ((h_s0_x_loop_min + h_s0_x_loop_extent) + -1))}",
                        (coli::expr((uint8_t)2) * input(coli::idx("h_s0_c"), coli::idx("h_s0_y"), coli::idx("h_s0_x"))), true, coli::p_uint8, &fusion_coli);
    h_s0.set_access("{h_s0[h_s0_c, h_s0_y, h_s0_x]->buff_h[h_s0_c, h_s0_y, h_s0_x]}");



    coli::constant k_s0_c_loop_min("k_s0_c_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &fusion_coli);
    coli::constant k_s0_c_loop_extent("k_s0_c_loop_extent", coli::expr(f_extent_2), coli::p_int32, true, NULL, 0, &fusion_coli);
    coli::constant k_s0_y_loop_min("k_s0_y_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &fusion_coli);
    coli::constant k_s0_y_loop_extent("k_s0_y_loop_extent", coli::expr(f_extent_1), coli::p_int32, true, NULL, 0, &fusion_coli);
    coli::constant k_s0_x_loop_min("k_s0_x_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &fusion_coli);
    coli::constant k_s0_x_loop_extent("k_s0_x_loop_extent", coli::expr(f_extent_0), coli::p_int32, true, NULL, 0, &fusion_coli);
    coli::computation k_s0("[k_s0_c_loop_min, k_s0_c_loop_extent, k_s0_y_loop_min, k_s0_y_loop_extent, k_s0_x_loop_min, k_s0_x_loop_extent]->{k_s0[k_s0_c, k_s0_y, k_s0_x]: "
                         "(k_s0_c_loop_min <= k_s0_c <= ((k_s0_c_loop_min + k_s0_c_loop_extent) + -1)) and (k_s0_y_loop_min <= k_s0_y <= ((k_s0_y_loop_min + k_s0_y_loop_extent) + -1)) and (k_s0_x_loop_min <= k_s0_x <= ((k_s0_x_loop_min + k_s0_x_loop_extent) + -1))}",
                         (coli::expr((uint8_t)2) * input(coli::idx("k_s0_c"), coli::idx("k_s0_y"), coli::idx("k_s0_x"))), true, coli::p_uint8, &fusion_coli);
    k_s0.set_access("{k_s0[k_s0_c, k_s0_y, k_s0_x]->buff_k[k_s0_c, k_s0_y, k_s0_x]}");



    fusion_coli.set_context_set("[f_s0_c_loop_min, f_s0_c_loop_extent, f_s0_y_loop_min, f_s0_y_loop_extent, f_s0_x_loop_min, f_s0_x_loop_extent, g_s0_c_loop_min, g_s0_c_loop_extent, g_s0_y_loop_min, g_s0_y_loop_extent, g_s0_x_loop_min, g_s0_x_loop_extent, h_s0_c_loop_min, h_s0_c_loop_extent, h_s0_y_loop_min, h_s0_y_loop_extent, h_s0_x_loop_min, h_s0_x_loop_extent, k_s0_c_loop_min, k_s0_c_loop_extent, k_s0_y_loop_min, k_s0_y_loop_extent, k_s0_x_loop_min, k_s0_x_loop_extent]->{: k_s0_x_loop_extent=f_s0_x_loop_extent and k_s0_x_loop_min=f_s0_x_loop_min and k_s0_y_loop_extent=f_s0_y_loop_extent and k_s0_y_loop_min=f_s0_y_loop_min and k_s0_c_loop_extent=f_s0_c_loop_extent and k_s0_c_loop_min=f_s0_c_loop_min and         h_s0_x_loop_extent=f_s0_x_loop_extent and h_s0_x_loop_min=f_s0_x_loop_min and h_s0_y_loop_extent=f_s0_y_loop_extent and h_s0_y_loop_min=f_s0_y_loop_min and h_s0_c_loop_extent=f_s0_c_loop_extent and f_s0_c_loop_min=h_s0_c_loop_min and f_s0_c_loop_min=g_s0_c_loop_min and f_s0_c_loop_extent=g_s0_c_loop_extent and f_s0_y_loop_min=g_s0_y_loop_min and f_s0_y_loop_extent=g_s0_y_loop_extent and f_s0_c_loop_min>=0 and f_s0_c_loop_extent>0}");

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


    fusion_coli.set_arguments({&buff_input, &buff_f, &buff_g, &buff_h, &buff_k});
    fusion_coli.gen_time_processor_domain();
    fusion_coli.gen_isl_ast();
    fusion_coli.gen_halide_stmt();
    fusion_coli.dump_halide_stmt();
    fusion_coli.gen_halide_obj("build/generated_fct_fusion.o");

    return 0;
}

