#include "Halide.h"

#define CV_DESCALE(x,n) (((x) + (1 << ((n)-1))) >> (n))

using namespace Halide;


int main(int argc, char* argv[]) {
    ImageParam in{UInt(32), 3, "input"};

    Func RGB2Gray{"RGB2Gray"};
    Var x,y,c;

    const Expr yuv_shift = cast<uint32_t>(14);
    const Expr R2Y = cast<uint32_t>(4899);
    const Expr G2Y = cast<uint32_t>(9617);
    const Expr B2Y = cast<uint32_t>(1868);


    RGB2Gray(x, y) = cast<uint8_t>(CV_DESCALE( (in(x, y, 2) * B2Y
                                + in(x, y, 1) * G2Y
                                + in(x, y, 0) * R2Y),
                                yuv_shift));

    RGB2Gray.parallel(y).vectorize(x, 8);
    //in.set_stride(0, 3)  // stride in dimension 0 (x) is three
    //  .set_stride(2, 1); // stride in dimension 2 (c) is one
    in.set_bounds(2, 0, 3);

    RGB2Gray.compile_to_object("build/generated_fct_cvtcolor_ref.o", {in}, "cvtcolor_ref");

    RGB2Gray.compile_to_lowered_stmt("build/generated_fct_cvtcolor_ref.txt", {in}, Text);

    return 0;
}

