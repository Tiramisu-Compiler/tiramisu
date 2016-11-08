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

    Halide::Image<uint8_t> in_image = Halide::Tools::load_image("./images/rgb.png");
    int SIZE0 = in_image.extent(0);
    int SIZE1 = in_image.extent(1);

    coli::function cvtcolor_coli("cvtcolor_coli");

    // Output buffers.
    int RGB2Gray_extent_1 = SIZE1;
    int RGB2Gray_extent_0 = SIZE0;
    coli::buffer buff_RGB2Gray("buff_RGB2Gray", 2, {coli::expr(RGB2Gray_extent_1), coli::expr(RGB2Gray_extent_0)}, coli::p_uint8, NULL, coli::a_output, &cvtcolor_coli);

    // Input buffers.
    int input_extent_2 = 3;
    int input_extent_1 = SIZE1;
    int input_extent_0 = SIZE0;
    coli::buffer buff_input("buff_input", 3, {coli::expr(input_extent_2), coli::expr(input_extent_1), coli::expr(input_extent_0)}, coli::p_uint8, NULL, coli::a_input, &cvtcolor_coli);
    coli::computation input("[input_extent_2, input_extent_1, input_extent_0]->{input[i2, i1, i0]: (0 <= i2 <= (input_extent_2 + -1)) and (0 <= i1 <= (input_extent_1 + -1)) and (0 <= i0 <= (input_extent_0 + -1))}", expr(), false, coli::p_uint8, &cvtcolor_coli);
    input.set_access("{input[i2, i1, i0]->buff_input[i2, i1, i0]}");


    // Define loop bounds for dimension "RGB2Gray_s0_v4".
    coli::constant RGB2Gray_s0_v4_loop_min("RGB2Gray_s0_v4_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &cvtcolor_coli);
    coli::constant RGB2Gray_s0_v4_loop_extent("RGB2Gray_s0_v4_loop_extent", coli::expr(RGB2Gray_extent_1), coli::p_int32, true, NULL, 0, &cvtcolor_coli);

    // Define loop bounds for dimension "RGB2Gray_s0_v3".
    coli::constant RGB2Gray_s0_v3_loop_min("RGB2Gray_s0_v3_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &cvtcolor_coli);
    coli::constant RGB2Gray_s0_v3_loop_extent("RGB2Gray_s0_v3_loop_extent", coli::expr(RGB2Gray_extent_0), coli::p_int32, true, NULL, 0, &cvtcolor_coli);
    coli::computation RGB2Gray_s0("[RGB2Gray_s0_v4_loop_min, RGB2Gray_s0_v4_loop_extent, RGB2Gray_s0_v3_loop_min, RGB2Gray_s0_v3_loop_extent]->{RGB2Gray_s0[RGB2Gray_s0_v4, RGB2Gray_s0_v3]: "
                        "(RGB2Gray_s0_v4_loop_min <= RGB2Gray_s0_v4 <= ((RGB2Gray_s0_v4_loop_min + RGB2Gray_s0_v4_loop_extent) + -1)) and (RGB2Gray_s0_v3_loop_min <= RGB2Gray_s0_v3 <= ((RGB2Gray_s0_v3_loop_min + RGB2Gray_s0_v3_loop_extent) + -1))}",
                        coli::expr(coli::o_cast, coli::p_uint8, (((((coli::expr(coli::o_cast, coli::p_uint32, input(coli::expr((int32_t)2), coli::idx("RGB2Gray_s0_v4"), coli::idx("RGB2Gray_s0_v3")))*coli::expr((uint32_t)1868)) + (coli::expr(coli::o_cast, coli::p_uint32, input(coli::expr((int32_t)1), coli::idx("RGB2Gray_s0_v4"), coli::idx("RGB2Gray_s0_v3")))*coli::expr((uint32_t)9617))) + (coli::expr(coli::o_cast, coli::p_uint32, input(coli::expr((int32_t)0), coli::idx("RGB2Gray_s0_v4"), coli::idx("RGB2Gray_s0_v3")))*coli::expr((uint32_t)4899))) + (coli::expr((uint32_t)1) << (coli::expr((uint32_t)14) - coli::expr((uint32_t)1)))) >> coli::expr((uint32_t)14))), true, coli::p_uint8, &cvtcolor_coli);
    RGB2Gray_s0.set_access("{RGB2Gray_s0[RGB2Gray_s0_v4, RGB2Gray_s0_v3]->buff_RGB2Gray[RGB2Gray_s0_v4, RGB2Gray_s0_v3]}");

    // Define compute level for "RGB2Gray".
    RGB2Gray_s0.first(computation::root_dimension);

    // Add schedules.
    RGB2Gray_s0.tag_parallel_dimension(0);

    cvtcolor_coli.set_arguments({&buff_input, &buff_RGB2Gray});
    cvtcolor_coli.gen_time_processor_domain();
    cvtcolor_coli.gen_isl_ast();
    cvtcolor_coli.gen_halide_stmt();
    cvtcolor_coli.dump_halide_stmt();
    cvtcolor_coli.gen_halide_obj("build/generated_fct_cvtcolor.o");

    return 0;
}

