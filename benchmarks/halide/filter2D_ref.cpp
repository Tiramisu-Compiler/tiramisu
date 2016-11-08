#include "Halide.h"
#include "wrapper_filter2D.h"

using namespace Halide;

int main(int argc, char* argv[]) {
    ImageParam in(UInt(8), 3, "input");
    ImageParam kernel(Float(32), 2, "kernel");

    Func filter2D("filter2D");
    Var x("x"), y("y"), c("c");

    Expr e = 0.0f;
    for (int j = 0; j < 3; j++) {
        for (int i = 0; i < 3; i++) {
            e += cast<float>(in(x + i, y + j, c)) * kernel(i, j);
        }
    }

    filter2D(x, y, c) = cast<uint8_t>(e);

    filter2D.parallel(y);//.vectorize(x, 8);

    filter2D.compile_to_object("build/generated_fct_filter2D_ref.o", {in, kernel}, "filter2D_ref");

    filter2D.compile_to_lowered_stmt("build/generated_fct_filter2D_ref.txt", {in, kernel}, Text);

    return 0;
}
