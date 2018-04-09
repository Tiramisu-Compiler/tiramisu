//
// Created by malek on 3/6/18.
//

#include "Halide.h"

#define CV_DESCALE(x,n) (((x) + (1 << ((n)-1))) >> (n))

using namespace Halide;

int main(int argc, char* argv[]) {
    ImageParam in{UInt(8), 3, "input"};

    Func RGB2Gray{"RGB2Gray"};
    Var x, y, c, x0, y0, x1, y1;

    const Expr yuv_shift = cast<uint32_t>(14);
    const Expr R2Y = cast<uint32_t>(4899);
    const Expr G2Y = cast<uint32_t>(9617);
    const Expr B2Y = cast<uint32_t>(1868);

    RGB2Gray(x, y) = cast<uint8_t>(CV_DESCALE( (in(x, y, 2) * B2Y
                                                + in(x, y, 1) * G2Y
                                                + in(x, y, 0) * R2Y),
                                               yuv_shift));

    RGB2Gray.compute_root();
    RGB2Gray.gpu_tile(x, y, x0, y0, x1, y1, 16, 16);

    Halide::Target target = Halide::get_host_target();
    target.set_feature(Target::Feature::CUDA, true);

    RGB2Gray.compile_to_object("build/generated_fct_cvtcolorgpu_ref.o", {in}, "cvtcolorgpu_ref", target);

    RGB2Gray.compile_to_lowered_stmt("build/generated_fct_cvtcolorgpu_ref.txt", {in}, Text, target);

    return 0;
}
