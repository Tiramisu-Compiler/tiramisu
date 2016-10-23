#include "Halide.h"
using namespace Halide;

int main(int argc, char **argv) {

    ImageParam rgb(UInt(8), 2);

    Var x("x"), y("y"), z("z");
    Func y_part("y_part"), u_part("u_part"), v_part("v_part");

    y_part(x, y) = ((66 * rgb(x, y, 0) + 129 * rgb(x, y, 1) +  25 * rgb(x, y, 2) + 128) >> 8) +  16;
    u_part(x, y) = (( -38 * rgb(2*x, 2*y, 0) -  74 * rgb(2*x, 2*y, 1) + 112 * rgb(2*x, 2*y, 2) + 128) >> 8) + 128;
    v_part(x, y) = (( 112 * rgb(2*x, 2*y, 0) -  94 * rgb(2*x, 2*y, 1) -  18 * rgb(2*x, 2*y, 2) + 128) >> 8) + 128;

    y_part.realize();
    u_part.realize();
    v_part.realize();

    // u_part.compute_with(y_part, y);
    // v_part.compute_with(u_part, y);
    // rgb.compute_at(y_part, y);

    Pipeline({y_part, u_part, v_part}).realize({y_im, u_im, v_im});

    Halide::Target target = Halide::get_host_target();
    Pipeline({y_part, u_part, v_part}).compile_to_object("build/generated_fct_rgb2yuv_ref.o",
                                                         {rgb},
                                                         "rgb2yuv_ref",
                                                         target);

    Pipeline({y_part, u_part, v_part}).compile_to_lowered_stmt("build/generated_fct_rgb2yuv_ref.txt",
                                                               {rgb},
                                                               Text,
                                                               target);

    return 0;
}
