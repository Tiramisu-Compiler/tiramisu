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

/**
 * Test the command compute_at().
 */

#define USE_FIXED_VALUES 0
#define SEPARATED 0

using namespace tiramisu;

int main(int argc, char **argv)
{
    global::set_default_tiramisu_options();

    tiramisu::function blurxy_tiramisu("blurxy_tiramisu_test");


    Halide::Buffer<uint8_t> in_image = Halide::Tools::load_image("./images/rgb.png");
    int SIZE0 = in_image.extent(0);
    int SIZE1 = in_image.extent(1);
    int SIZE2 = in_image.extent(2);


    int by_ext_2 = SIZE2;
    int by_ext_1 = SIZE1 - 8;
    int by_ext_0 = SIZE0 - 8;

    std::cout << "Nc = " << by_ext_2 << ", Ny = " << by_ext_1 + 2 << ", Nx = " << by_ext_0 << std::endl;
    std::cout << "Mc = " << by_ext_2 << ", My = " << by_ext_1 << ", Mx = " << by_ext_0 << std::endl;

    tiramisu::buffer buff_p0("buff_p0", 3, {tiramisu::expr(SIZE2), tiramisu::expr(SIZE1), tiramisu::expr(SIZE0)},
                             tiramisu::p_uint8, NULL, tiramisu::a_input, &blurxy_tiramisu);
    tiramisu::buffer buff_bx("buff_bx", 3, {tiramisu::expr(by_ext_2), tiramisu::expr(by_ext_1 + 2), tiramisu::expr(by_ext_0)},
                             tiramisu::p_uint8, NULL, tiramisu::a_temporary, &blurxy_tiramisu);
    tiramisu::buffer buff_by("buff_by", 3, {tiramisu::expr(by_ext_2), tiramisu::expr(by_ext_1), tiramisu::expr(by_ext_0)},
                             tiramisu::p_uint8, NULL, tiramisu::a_output, &blurxy_tiramisu);


    tiramisu::computation p0("[SIZE2, SIZE1, SIZE0]->{p0[i2, i1, i0]: (0 <= i2 <= (SIZE2 -1)) and (0 <= i1 <= (SIZE1 -1)) and (0 <= i0 <= (SIZE0 -1))}",
       expr(), false, tiramisu::p_uint8, &blurxy_tiramisu);
    p0.set_access("{p0[i2, i1, i0]->buff_p0[i2, i1, i0]}");

    tiramisu::constant Nc("Nc", tiramisu::expr(by_ext_2), tiramisu::p_int32, true, NULL, 0,
                          &blurxy_tiramisu);
    tiramisu::constant Ny("Ny", (tiramisu::expr(by_ext_1) + tiramisu::expr((int32_t)2)),
                          tiramisu::p_int32, true, NULL, 0, &blurxy_tiramisu);
    tiramisu::constant Nx("Nx", tiramisu::expr(by_ext_0), tiramisu::p_int32, true, NULL, 0,
                          &blurxy_tiramisu);
    tiramisu::computation
    bx("[Nc, Ny, Nx]->{bx[c, y, x]: (0 <= c <= (Nc -1)) and (0 <= y <= (Ny -1)) and (0 <= x <= (Nx -1))}",
       (((p0(tiramisu::var("c"), tiramisu::var("y"), tiramisu::var("x")) + p0(tiramisu::var("c"),
               tiramisu::var("y"), (tiramisu::var("x") + tiramisu::expr((int32_t)1)))) + p0(tiramisu::var("c"),
                       tiramisu::var("y"), (tiramisu::var("x") + tiramisu::expr((int32_t)2)))) / tiramisu::expr((
                                   uint8_t)3)), true, tiramisu::p_uint8, &blurxy_tiramisu);
    bx.set_access("{bx[c, y, x]->buff_bx[c, y, x]}");


    tiramisu::constant Mc("Mc", tiramisu::expr(by_ext_2), tiramisu::p_int32, true, NULL, 0,
                          &blurxy_tiramisu);
    tiramisu::constant My("My", tiramisu::expr(by_ext_1), tiramisu::p_int32, true, NULL, 0,
                          &blurxy_tiramisu);
    tiramisu::constant Mx("Mx", tiramisu::expr(by_ext_0), tiramisu::p_int32, true, NULL, 0,
                          &blurxy_tiramisu);
    tiramisu::computation
    by("[Mc, My, Mx]->{by[c, y, x]: (0 <= c <= (Mc -1)) and (0 <= y <= (My -1)) and (0 <= x <= (Mx -1))}",
       (((bx(tiramisu::var("c"), tiramisu::var("y"), tiramisu::var("x")) + bx(tiramisu::var("c"),
               (tiramisu::var("y") + tiramisu::expr((int32_t)1)), tiramisu::var("x"))) + bx(tiramisu::var("c"),
                       (tiramisu::var("y") + tiramisu::expr((int32_t)2)),
                       tiramisu::var("x"))) / tiramisu::expr((uint8_t)3)), true, tiramisu::p_uint8, &blurxy_tiramisu);
    by.set_access("{by[c, y, x]->buff_by[c, y, x]}");


#if USE_FIXED_VALUES
    blurxy_tiramisu.add_context_constraints("[Nc, Ny, Nx, Mc, My, Mx]->{: Nc=3 and Mc=3 and Ny=3514 and My=3512 and Nx=2104 and Mx=2104}");
#else
    blurxy_tiramisu.add_context_constraints("[Nc, Ny, Nx, Mc, My, Mx]->{: Nc=Mc and Ny>My and Nx=Mx and Nc>0 and Ny>0 and Nx>0 and Mc>0 and My>0 and Mx>0}");
#endif


#if 0
    // Default schedule.
    bx.set_schedule("[Nc, Ny, Nx]->{bx[c,y,x]->bx[0, 0, c, 0, y, 0, x, 0]: (0 <= c <= (Nc -1)) and (0 <= y <= (Ny -1)) and (0 <= x <= (Nx -1))}");
    by.set_schedule("[Mc, My, Mx]->{by[c,y,x]->by[0, 1, c, 0, y, 0, x, 0]: (0 <= c <= (Mc -1)) and (0 <= y <= (My -1)) and (0 <= x <= (Mx -1))}");
#elif 0
    bx.set_schedule("[Nc, Ny, Nx]->{bx[c,y,x]->bx[0, 0, c, 0, floor(y/32), 0, floor(x/32), 0, (y%32), 0, (x%32), 0]: (0 <= c <= (Nc -1)) and (0 <= y <= (Ny -1)) and (0 <= x <= (Nx -1)); bx [c,y,x]->bx [1, 0, c, 0, floor((y-2)/32), 0, floor(x/32), 1, ((y-2)%32), 0, (x%32), 0]: (0 <= c <= (Nc -1)) and ((0 <= (y%32) <= 2)) and (y>=2) and (0 <= y <= (Ny -1)) and (0 <= x <= (Nx -1))}");
    by.set_schedule("[Mc, My, Mx]->{by[c,y,x]->by[0, 0, c, 0, floor(y/32), 0, floor(x/32), 2, (y%32), 0, (x%32), 0]: (0 <= c <= (Mc -1)) and (0 <= y <= (My -1)) and (0 <= x <= (Mx -1))}");
#elif 0
    // duplicate
    bx.create_duplication_transformation("[Nc, Ny, Nx]->{bx[c, y, x]->bx[1, 0, c, 0, y, 0, x, 0]: (0 <= c <= (Nc -1)) and ((0 <= (y%32) <= 2)) and (y>=2) and (0 <= y <= (Ny -1)) and (0 <= x <= (Nx -1))}");
    // tiling of the original
    bx.apply_transformation("[Nc, Ny, Nx]->{bx[0, 0, c, 0, y, 0, x, 0]->bx[0, 0, c, 0, floor(y/32), 0, floor(x/32), 0, (y%32), 0, (x%32), 0]}",
                            0);
    //  shift the duplicate here
    bx.apply_transformation("[Nc, Ny, Nx]->{bx[1, 0, c, 0, y, 0, x, 0]->bx[1, 0, c, 0, y-2, 0, x, 0]}",
                            1);
    // tiling of the duplicate
    bx.apply_transformation("[Nc, Ny, Nx]->{bx[1, 0, c, 0, y, 0, x, 0]->bx[1, 0, c, 0, floor(y/32), 0, floor(x/32), 0, (y%32), 0, (x%32), 0]}",
                            1);
    // duplicate 1 after the original of bx.
    bx.apply_transformation("[Nc, Ny, Nx]->{bx[1, 0, c, 0, y1, 0, x1, 0, y2, 0, x2, 0]->bx[1, 0, c, 0, y1, 0, x1, 1, y2, 0, x2, 0]}",
                            1);
    // tile by
    by.apply_transformation("[Mc, My, Mx]->{by[0, 0, c, 0, y, 0, x, 0]->by[0, 0, c, 0, floor(y/32), 0, floor(x/32), 0, (y%32), 0, (x%32), 0]}",
                            0);
    // by after bx
    by.apply_transformation("[Mc, My, Mx]->{by[0, 0, c, 0, y1, 0, x1, 0, y2, 0, x2, 0]->by[0, 0, c, 0, y1, 0, x1, 2, y2, 0, x2, 0]}",
                            0);
#elif 0
    bx.duplicate("[Nc, Ny, Nx]->{bx[c, y, x]: (0 <= c <= (Nc -1)) and (0 <= y <= (Ny -1)) and (0 <= (y%32) <= 2) and (y>=2) and (0 <= x <= (Nx -1))}",
                 "");
    bx.select(1)->shift(1, -2);
    bx.tile(1, 2, 32, 32);
    bx.select(1)->tile(1, 2, 32, 32);
    by.tile(1, 2, 32, 32);
    bx.select(1)->after(bx, 2, 0);
    by.after(bx, 2, 1);
#elif 1
    bx.tile(1, 2, 32, 32);
    by.tile(1, 2, 32, 32);
    bx.compute_at(by, 2);

#if SEPARATED || USE_FIXED_VALUES
    bx.after(bx, computation::root_dimension, 1);
    by.after(bx, computation::root_dimension, 1);
    by.after(bx, 2, 0);
#endif
#endif

    bx.tag_parallel_level(1);
    bx.tag_parallel_level(2);
    by.tag_parallel_level(1);
    by.tag_parallel_level(2);

    blurxy_tiramisu.set_arguments({&buff_p0, &buff_by});
    blurxy_tiramisu.gen_time_processor_domain();

    blurxy_tiramisu.gen_isl_ast();
    blurxy_tiramisu.gen_halide_stmt();
    blurxy_tiramisu.dump_halide_stmt();
    blurxy_tiramisu.gen_c_code();
    blurxy_tiramisu.gen_halide_obj("build/generated_fct_test_14.o");
    blurxy_tiramisu.gen_c_code();
    blurxy_tiramisu.dump_schedule();

    return 0;
}
