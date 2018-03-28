#include "Halide.h"

using namespace Halide;

#define kernelX_length 7
#define kernelY_length 7

int main(int argc, char* argv[]) {
    ImageParam in(UInt(8), 3, "input");
    ImageParam kernelX(Float(32), 1, "kernelx");
    ImageParam kernelY(Float(32), 1, "kernely");

    Func gaussian_gpu("gaussian_gpu"), gaussian_gpu_x("gaussian_gpu_x");
    Var x("x"), y("y"), c("c");

    Expr e = 0.0f;
    for (int i = 0; i < 5; ++i) {
        e += in(x + i, y, c) * kernelX(i);
    }
    gaussian_gpu_x(x, y, c) = e;

    Expr f = 0.0f;
    for (int j = 0; j < 5; ++j) {
        f += gaussian_gpu_x(x, y + j, c) * kernelY(j);
    }
    gaussian_gpu(x, y, c) = cast<uint8_t>(f);

    Var x_inner, y_inner, x_outer, y_outer, tile_index;
//    gaussian_gpu.tile(x, y, x_outer, y_outer, x_inner, y_inner, 8, 8)
//            .fuse(x_outer, y_outer, tile_index)
//            .compute_root();
//            .parallel(x_outer);
    gaussian_gpu_x.compute_root();

    gaussian_gpu_x.gpu_tile(x, y, x_inner, y_inner, x_outer, y_outer, 16, 16);
    gaussian_gpu.gpu_tile(x, y, x_inner, y_inner, x_outer, y_outer, 16, 16);

    Halide::Target target = Halide::get_host_target();
    target.set_feature(Target::Feature::CUDA, true);

    gaussian_gpu.compile_to_object("build/generated_fct_gaussian_gpu_ref.o", {in, kernelX, kernelY}, "gaussian_gpu_ref", target);

    gaussian_gpu.compile_to_lowered_stmt("build/generated_fct_gaussian_gpu_ref.txt", {in, kernelX, kernelY}, Text, target);

  return 0;
}

