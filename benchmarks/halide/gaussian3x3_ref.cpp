#include "Halide.h"

using namespace Halide;

#define kernelX_length 7
#define kernelY_length 7

int main(int argc, char* argv[]) {
    ImageParam in{Float(32), 2, "input"};
    ImageParam kernelX{Float(32), 1, "kernelx"};
    ImageParam kernelY{Float(32), 1, "kernely"};

    Func gaussian{"gaussian"};
    Func gaussian_x{"gaussian_x"};
    Var x,y;

    Expr e,f;
    e = 0.0f;
    for (int i=0; i<kernelX_length; i++) {
        e += in(x+i,y) * kernelX(i);
    }
    gaussian_x(x, y) = e;

    f = 0.0f;
    for (int i=0; i<kernelX_length; i++) {
        f += gaussian_x(x, y+i) * kernelY(i);
    }

    gaussian(x, y) = f;

    Var x_inner, y_inner, x_outer, y_outer, tile_index;
    gaussian.tile(x, y, x_outer, y_outer, x_inner, y_inner, 4, 4)
            .fuse(x_outer, y_outer, tile_index).compute_root().parallel(tile_index);
    gaussian_x.compute_at(gaussian, y_inner);

    gaussian.compile_to_object("build/generated_fct_gaussian_3x3_ref.o", {in, kernelX, kernelY}, "gaussian_3x3_ref");

    gaussian.compile_to_lowered_stmt("build/generated_fct_gaussian_3x3_ref.txt", {in, kernelX, kernelY}, Text);

  return 0;
}

