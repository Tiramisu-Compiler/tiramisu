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

    std::cout << "INPUT WIDTH: " << in_image.width() << "\n";
    std::cout << "INPUT HEIGHT: " << in_image.height() << "\n";
    std::cout << "INPUT CHANNELS: " << in_image.channels() << "\n";

    // Output buffers.
    int f_extent_2 = SIZE2;
    int f_extent_1 = SIZE1;
    int f_extent_0 = SIZE0;
    coli::buffer buff_f("buff_f", 3, {coli::expr(f_extent_2), coli::expr(f_extent_1), coli::expr(f_extent_0)}, coli::p_uint8, NULL, coli::a_output, &fusion_coli);
    int g_extent_2 = SIZE2;
    int g_extent_1 = SIZE1;
    int g_extent_0 = SIZE0;
    coli::buffer buff_g("buff_g", 3, {coli::expr(g_extent_2), coli::expr(g_extent_1), coli::expr(g_extent_0)}, coli::p_uint8, NULL, coli::a_output, &fusion_coli);

    // Input buffers.
    int input_extent_2 = SIZE2;
    int input_extent_1 = SIZE1;
    int input_extent_0 = SIZE0;
    coli::buffer buff_input("buff_input", 3, {coli::expr(input_extent_2), coli::expr(input_extent_1), coli::expr(input_extent_0)}, coli::p_uint8, NULL, coli::a_input, &fusion_coli);
    coli::computation input("[input_extent_2, input_extent_1, input_extent_0]->{input[i2, i1, i0]: (0 <= i2 <= (input_extent_2 + -1)) and (0 <= i1 <= (input_extent_1 + -1)) and (0 <= i0 <= (input_extent_0 + -1))}", expr(), false, coli::p_uint8, &fusion_coli);
    input.set_access("{input[i2, i1, i0]->buff_input[i2, i1, i0]}");


    // Define loop bounds for dimension "f_s0_c".
    coli::constant f_s0_c_loop_min("f_s0_c_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &fusion_coli);
    coli::constant f_s0_c_loop_extent("f_s0_c_loop_extent", coli::expr(f_extent_2), coli::p_int32, true, NULL, 0, &fusion_coli);

    // Define loop bounds for dimension "f_s0_y".
    coli::constant f_s0_y_loop_min("f_s0_y_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &fusion_coli);
    coli::constant f_s0_y_loop_extent("f_s0_y_loop_extent", coli::expr(f_extent_1), coli::p_int32, true, NULL, 0, &fusion_coli);

    // Define loop bounds for dimension "f_s0_x".
    coli::constant f_s0_x_loop_min("f_s0_x_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &fusion_coli);
    coli::constant f_s0_x_loop_extent("f_s0_x_loop_extent", coli::expr(f_extent_0), coli::p_int32, true, NULL, 0, &fusion_coli);
    coli::computation f_s0("[f_s0_c_loop_min, f_s0_c_loop_extent, f_s0_y_loop_min, f_s0_y_loop_extent, f_s0_x_loop_min, f_s0_x_loop_extent]->{f_s0[f_s0_c, f_s0_y, f_s0_x]: "
                        "(f_s0_c_loop_min <= f_s0_c <= ((f_s0_c_loop_min + f_s0_c_loop_extent) + -1)) and (f_s0_y_loop_min <= f_s0_y <= ((f_s0_y_loop_min + f_s0_y_loop_extent) + -1)) and (f_s0_x_loop_min <= f_s0_x <= ((f_s0_x_loop_min + f_s0_x_loop_extent) + -1))}",
                        (coli::expr((uint8_t)255) - input(coli::idx("f_s0_c"), coli::idx("f_s0_y"), coli::idx("f_s0_x"))), true, coli::p_uint8, &fusion_coli);
    f_s0.set_access("{f_s0[f_s0_c, f_s0_y, f_s0_x]->buff_f[f_s0_c, f_s0_y, f_s0_x]}");

    // Define compute level for "f".
    f_s0.first(computation::root_dimension);

    // Define loop bounds for dimension "g_s0_c".
    coli::constant g_s0_c_loop_min("g_s0_c_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &fusion_coli);
    coli::constant g_s0_c_loop_extent("g_s0_c_loop_extent", coli::expr(g_extent_2), coli::p_int32, true, NULL, 0, &fusion_coli);

    // Define loop bounds for dimension "g_s0_y".
    coli::constant g_s0_y_loop_min("g_s0_y_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &fusion_coli);
    coli::constant g_s0_y_loop_extent("g_s0_y_loop_extent", coli::expr(g_extent_1), coli::p_int32, true, NULL, 0, &fusion_coli);

    // Define loop bounds for dimension "g_s0_x".
    coli::constant g_s0_x_loop_min("g_s0_x_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &fusion_coli);
    coli::constant g_s0_x_loop_extent("g_s0_x_loop_extent", coli::expr(g_extent_0), coli::p_int32, true, NULL, 0, &fusion_coli);
    coli::computation g_s0("[g_s0_c_loop_min, g_s0_c_loop_extent, g_s0_y_loop_min, g_s0_y_loop_extent, g_s0_x_loop_min, g_s0_x_loop_extent]->{g_s0[g_s0_c, g_s0_y, g_s0_x]: "
                        "(g_s0_c_loop_min <= g_s0_c <= ((g_s0_c_loop_min + g_s0_c_loop_extent) + -1)) and (g_s0_y_loop_min <= g_s0_y <= ((g_s0_y_loop_min + g_s0_y_loop_extent) + -1)) and (g_s0_x_loop_min <= g_s0_x <= ((g_s0_x_loop_min + g_s0_x_loop_extent) + -1))}",
                        (coli::expr((uint8_t)2) * input(coli::idx("g_s0_c"), coli::idx("g_s0_y"), coli::idx("g_s0_x"))), true, coli::p_uint8, &fusion_coli);
    g_s0.set_access("{g_s0[g_s0_c, g_s0_y, g_s0_x]->buff_g[g_s0_c, g_s0_y, g_s0_x]}");

    // Define compute level for "g".
    g_s0.after(f_s0, computation::root_dimension);

    // Add schedules.

    fusion_coli.set_arguments({&buff_input, &buff_f, &buff_g});
    fusion_coli.gen_time_processor_domain();
    fusion_coli.gen_isl_ast();
    fusion_coli.gen_halide_stmt();
    fusion_coli.dump_halide_stmt();
    fusion_coli.gen_halide_obj("build/generated_fct_fusion.o");

    return 0;
}

