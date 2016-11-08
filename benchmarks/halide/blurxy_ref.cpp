#include "Halide.h"
using namespace Halide;

int main(int argc, char **argv) {

    ImageParam input(UInt(16), 2);
    Func blur_x("blur_x"), blur_y("blur_y");
    Var x("x"), y("y"), xi("xi"), yi("yi");

    // The algorithm
    blur_x(x, y) = (input(x, y) + input(x+1, y) + input(x+2, y))/3;
    blur_y(x, y) = (blur_x(x, y) + blur_x(x, y+1) + blur_x(x, y+2))/3;

    // How to schedule it
    //blur_y.split(y, y, yi, 8).parallel(y).vectorize(x, 8);
    //blur_x.store_at(blur_y, y).compute_at(blur_y, yi).vectorize(x, 8);

    blur_y.compute_root().store_root().parallel(y);
    blur_x.compute_root().store_root().parallel(y);

    Halide::Target target = Halide::get_host_target();

    // blur_y.compile_to_coli("halide/wrap_affine/wrap_affine_algorithm.cpp",
    //                        {input}, target);

    blur_y.compile_to_object("build/generated_fct_blurxy_ref.o",
                             {input},
                             "blurxy_ref",
                             target);

    blur_y.compile_to_lowered_stmt("build/generated_fct_blurxy_ref.txt",
                                   {input},
                                   Text,
                                   target);

    return 0;
}
