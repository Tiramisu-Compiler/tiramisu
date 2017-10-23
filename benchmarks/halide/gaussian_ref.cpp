#include "Halide.h"

using namespace Halide;

#define kernelX_length 7
#define kernelY_length 7

int main(int argc, char* argv[]) {
    ImageParam in(UInt(8), 3, "input");
    ImageParam kernelX(Float(32), 1, "kernelx");
    ImageParam kernelY(Float(32), 1, "kernely");

    Func gaussian("gaussian"), gaussian_x("gaussian_x");
    Var x("x"), y("y"), c("c");

    Expr e = 0.0f;
    for (int i = 0; i < 5; ++i) {
        e += in(x + i, y, c) * kernelX(i);
    }
    gaussian_x(x, y, c) = e;

    Expr f = 0.0f;
    for (int j = 0; j < 5; ++j) {
        f += gaussian_x(x, y + j, c) * kernelY(j);
    }
    gaussian(x, y, c) = cast<uint8_t>(f);

    Var x_inner, y_inner, x_outer, y_outer, tile_index;
    gaussian.tile(x, y, x_outer, y_outer, x_inner, y_inner, 4, 4)
            .fuse(x_outer, y_outer, tile_index)
            .compute_root()
            .parallel(tile_index);
    gaussian_x.compute_at(gaussian, y_inner);

    gaussian.compile_to_object("build/generated_fct_gaussian_ref.o", {in, kernelX, kernelY}, "gaussian_ref");

    gaussian.compile_to_lowered_stmt("build/generated_fct_gaussian_ref.txt", {in, kernelX, kernelY}, Text);

  return 0;
}

