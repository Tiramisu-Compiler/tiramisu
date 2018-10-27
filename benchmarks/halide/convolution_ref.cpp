#include "Halide.h"
#include "wrapper_convolution.h"

using namespace Halide;

int main(int argc, char* argv[]) {
    ImageParam in(UInt(8), 3, "input");
    ImageParam kernel(Float(32), 2, "kernel");

    Func convolution("convolution");
    Var x("x"), y("y"), c("c");

    Expr e = 0.0f;
    for (int j = 0; j < 3; j++) {
        for (int i = 0; i < 3; i++) {
            e += cast<float>(in(x + i, y + j, c)) * kernel(i, j);
        }
    }

    convolution(x, y, c) = cast<uint8_t>(e);

    convolution.vectorize(x, 8).parallel(c);

    convolution.compile_to_object("build/generated_fct_convolution_ref.o", {in, kernel}, "convolution_ref");

    convolution.compile_to_lowered_stmt("build/generated_fct_convolution_ref.txt", {in, kernel}, Text);

    return 0;
}
