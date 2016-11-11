#include "Halide.h"
using namespace Halide;

int main(int argc, char **argv) {

    ImageParam input(UInt(8), 3);
    Func blur_x("blur_x"), blur_y("blur_y");
    Var x("x"), y("y"), c("c"), xi("xi"), yi("yi");

    // The algorithm
    blur_x(x, y, c) = (input(x, y, c) + input(x+1, y, c) + input(x+2, y, c))/3;
    blur_y(x, y, c) = (blur_x(x, y, c) + blur_x(x, y+1, c) + blur_x(x, y+2, c))/3;

    // How to schedule it
    blur_y.split(y, y, yi, 8).parallel(y).vectorize(x, 8).parallel(c);
    blur_x.store_at(blur_y, y).compute_at(blur_y, yi).vectorize(x, 8).parallel(c);

    Halide::Target target = Halide::get_host_target();

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
