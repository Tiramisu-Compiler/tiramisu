#include "Halide.h"

using namespace Halide;

#define kernelX_length 7
#define kernelY_length 7

int main(int argc, char* argv[]) {
    ImageParam in(UInt(8), 3, "input");
    ImageParam kernelX(Float(32), 1, "kernelx");
    ImageParam kernelY(Float(32), 1, "kernely");

    Func gaussiangpu("gaussiangpu"), gaussiangpu_x("gaussiangpu_x");
    Var x("x"), y("y"), c("c");

    Expr e = 0.0f;
    for (int i = 0; i < 5; ++i) {
        e += cast<float>(in(x + i, y, c))  * kernelX(i);
    }
    gaussiangpu_x(x, y, c) = e;

    Expr f = 0.0f;
     for (int j = 0; j < 5; ++j) {
         f += gaussiangpu_x(x, y + j, c) * kernelY(j);
     }
    gaussiangpu(x, y, c) = cast<uint8_t>(f);

    // gaussiangpu_x.reorder(c, x, y);
    gaussiangpu.reorder(c, x, y);

     Var x_inner, y_inner, x_outer, y_outer, tile_index;
// //    gaussiangpu.tile(x, y, x_outer, y_outer, x_inner, y_inner, 8, 8)
// //            .fuse(x_outer, y_outer, tile_index)
// //            .compute_root();
// //            .parallel(x_outer);
    // gaussiangpu_x.compute_root();
 
    // gaussiangpu_x.gpu_tile(x, y, x_inner, y_inner, x_outer, y_outer, 32, 32);
    gaussiangpu.gpu_tile(x, y, x_inner, y_inner, x_outer, y_outer, 32, 32);

    Halide::Target target = Halide::get_host_target();
    target.set_feature(Target::Feature::CUDA, true);

    gaussiangpu.compile_to_object("build/generated_fct_gaussiangpu_ref.o", {in, kernelX, kernelY}, "gaussiangpu_ref", target);

    gaussiangpu.compile_to_lowered_stmt("build/generated_fct_gaussiangpu_ref.txt", {in, kernelX, kernelY}, Text, target);

  return 0;
}

