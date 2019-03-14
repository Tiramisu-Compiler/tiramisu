#include "Halide.h"

#define CV_DESCALE(x,n) (((x) + (1 << ((n)-1))) >> (n))

using namespace Halide;


int main(int argc, char* argv[]) {
    ImageParam in{UInt(8), 3, "input"};

    Func RGB2Gray{"RGB2Gray"};
    Var x, y, c;

    const Expr yuv_shift = cast<uint32_t>(14);
    const Expr R2Y = cast<uint32_t>(4899);
    const Expr G2Y = cast<uint32_t>(9617);
    const Expr B2Y = cast<uint32_t>(1868);

    RGB2Gray(x, y) = cast<uint8_t>(CV_DESCALE( (in(x, y, 2) * B2Y
                                                + in(x, y, 1) * G2Y
                                                + in(x, y, 0) * R2Y),
                                               yuv_shift));

    RGB2Gray.parallel(y).vectorize(x, 8, Halide::TailStrategy::GuardWithIf);

    RGB2Gray.compile_to_object("/Users/jray/CLionProjects/tiramisu/build/generated_fct_cvtcolordist_ref.o", {in}, "cvtcolordist_ref");

    RGB2Gray.compile_to_lowered_stmt("/Users/jray/CLionProjects/tiramisu/build/generated_fct_cvtcolordist_ref.txt", {in}, Text);

    return 0;
}

