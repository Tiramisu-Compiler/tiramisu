#include "Halide.h"

using namespace Halide;

int main(int argc, char **argv) {

    ImageParam rgb(Int(32), 3);

    const int size = 1024;

    Image<int> y_im(size, size), u_im(size/2, size/2), v_im(size/2, size/2);
    //Image<int> y_im(rgb.width(), rgb.height()), u_im(rgb.width()/2, rgb.height()/2), v_im(rgb.width()/2, rgb.height()/2);

    Var x("x"), y("y");
    Func y_part("y_part"), u_part("u_part"), v_part("v_part");

    y_part(x, y) = ((66 * rgb(x, y, 0) + 129 * rgb(x, y, 1) +  25 * rgb(x, y, 2) + 128) >> 8) +  16;
    u_part(x, y) = (( -38 * rgb(2*x, 2*y, 0) -  74 * rgb(2*x, 2*y, 1) + 112 * rgb(2*x, 2*y, 2) + 128) >> 8) + 128;
    v_part(x, y) = (( 112 * rgb(2*x, 2*y, 0) -  94 * rgb(2*x, 2*y, 1) -  18 * rgb(2*x, 2*y, 2) + 128) >> 8) + 128;

    // u_part.compute_with(y_part, y);
    // v_part.compute_with(u_part, y);
    // rgb.compute_at(y_part, y);

    Halide::Target target = Halide::get_host_target();
    Pipeline({y_part, u_part, v_part}).compile_to_object("build/generated_fct_rgb2yuv_ref.o",
                                                         {rgb},
                                                         "rgb2yuv_ref",
                                                         target);

    Pipeline({y_part, u_part, v_part}).compile_to_lowered_stmt("build/generated_fct_rgb2yuv_ref.txt",
                                                               {rgb},
                                                               Halide::Text,
                                                               target);

    return 0;
}
