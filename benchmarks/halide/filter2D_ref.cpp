#include "Halide.h"
#include "wrapper_filter2D.h"

using namespace Halide;

int main(int argc, char* argv[]) {
    ImageParam in{Float(32), 2, "input"};
    // kernel is 2*radius x 2*radius
    ImageParam kernel{Float(32), 2, "kernel"};

    Func filter2D{"filter2D"};
    Var x, y;

    Expr e = 0.0f;

    for (int i=-RADIUS; i<RADIUS; i++) {
        for (int j=-RADIUS; j<RADIUS; j++)  {
        e += in(x+RADIUS+i, y+RADIUS+j) * kernel(RADIUS+i, RADIUS+j);
        }
    }

    filter2D(x, y) = e;

    filter2D.parallel(y).vectorize(x, 8);

    filter2D.compile_to_object("build/generated_fct_filter2D_ref.o", {in, kernel}, "filter2D_ref");

    filter2D.compile_to_lowered_stmt("build/generated_fct_filter2D_ref.txt", {in, kernel}, Text);

    return 0;
}
