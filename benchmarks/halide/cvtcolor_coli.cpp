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

    coli::function cvt_color_coli("cvtcolor_coli");
    coli::buffer buff_RGB2Gray("buff_RGB2Gray", 2, {coli::expr(100), coli::expr(100)}, coli::p_uint8, NULL, coli::a_output, &cvt_color_coli);
    coli::buffer buff_b0("buff_b0", 3, {coli::expr(100), coli::expr(100), coli::expr(3)}, coli::p_uint8, NULL, coli::a_input, &cvt_color_coli);
    coli::computation b0("{b0[i0, i1, i2]: (0 <= i0 <= 99) and (0 <= i1 <= 99) and (0 <= i2 <= 2)}", expr(), false, coli::p_uint8, &cvt_color_coli);
    b0.set_access("{b0[i0, i1, i2]->buff_b0[i0, i1, i2]}");

    // Define loop bounds for dimension "RGB2Gray_s0_y".
    coli::constant RGB2Gray_s0_y_loop_min("RGB2Gray_s0_y_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &cvt_color_coli);
    coli::constant RGB2Gray_s0_y_loop_extent("RGB2Gray_s0_y_loop_extent", coli::expr((int32_t)100), coli::p_int32, true, NULL, 0, &cvt_color_coli);

    // Define loop bounds for dimension "RGB2Gray_s0_x".
    coli::constant RGB2Gray_s0_x_loop_min("RGB2Gray_s0_x_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &cvt_color_coli);
    coli::constant RGB2Gray_s0_x_loop_extent("RGB2Gray_s0_x_loop_extent", coli::expr((int32_t)100), coli::p_int32, true, NULL, 0, &cvt_color_coli);
    coli::computation RGB2Gray_s0("[RGB2Gray_s0_y_loop_min, RGB2Gray_s0_y_loop_extent, RGB2Gray_s0_x_loop_min, RGB2Gray_s0_x_loop_extent]->{RGB2Gray_s0[RGB2Gray_s0_x, RGB2Gray_s0_y]: "
        "(RGB2Gray_s0_y_loop_min <= RGB2Gray_s0_y <= ((RGB2Gray_s0_y_loop_min + RGB2Gray_s0_y_loop_extent) + -1)) and (RGB2Gray_s0_x_loop_min <= RGB2Gray_s0_x <= ((RGB2Gray_s0_x_loop_min + RGB2Gray_s0_x_loop_extent) + -1))}",
        coli::expr(coli::o_cast, coli::p_uint8, (((((b0(coli::idx("RGB2Gray_s0_x"), coli::idx("RGB2Gray_s0_y"), coli::expr((uint8_t)2))*coli::expr((uint8_t)1868)) + (b0(coli::idx("RGB2Gray_s0_x"), coli::idx("RGB2Gray_s0_y"), coli::expr((uint8_t)1))*coli::expr((uint8_t)9617))) + (b0(coli::idx("RGB2Gray_s0_x"), coli::idx("RGB2Gray_s0_y"), coli::expr((uint8_t)0))*coli::expr((uint8_t)4899))) + coli::expr(coli::o_cast, coli::p_uint8, (coli::expr((uint8_t)1) << (coli::expr((uint8_t)14) - coli::expr((uint8_t)1))))) >> coli::expr((uint8_t)14))), true, coli::p_uint8, &cvt_color_coli);
    RGB2Gray_s0.set_access("{RGB2Gray_s0[RGB2Gray_s0_x, RGB2Gray_s0_y]->buff_RGB2Gray[RGB2Gray_s0_x, RGB2Gray_s0_y]}");

    // Define compute level for "RGB2Gray".
    RGB2Gray_s0.first(computation::root_dimension);

    // Add schedules.

    cvt_color_coli.set_arguments({&buff_b0, &buff_RGB2Gray});
    cvt_color_coli.gen_time_processor_domain();
    cvt_color_coli.gen_isl_ast();
    cvt_color_coli.gen_halide_stmt();
    cvt_color_coli.dump_halide_stmt();
    cvt_color_coli.gen_halide_obj("build/generated_fct_cvtcolor.o");

    return 0;
}
