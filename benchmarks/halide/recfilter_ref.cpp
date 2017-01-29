#include "Halide.h"

using namespace Halide;


int main(int argc, char* argv[]) {
    ImageParam in(UInt(8), 3, "input");

    float a0 = 0.7;
    float a1 = 0.2;
    float a2 = 0.1;

    Func rec_filter{"rec_filter"};
    Var x("x"), y("y"), c("c");
    RDom r(2, in.width()-3, 0, in.height()-1);

    rec_filter(x, y, c) = in(x, y, c);
    rec_filter(r.x, r.y, c) = cast<uint8_t>(a0*rec_filter(r.x, r.y, c) + a1*rec_filter(r.x-1, r.y, c) + a2*rec_filter(r.x-2, r.y, c));

    rec_filter.parallel(y).parallel(c).vectorize(x, 8);
    rec_filter.update(0).parallel(r.y).parallel(c);

    rec_filter.compile_to_object("build/generated_fct_recfilter_ref.o", {in}, "recfilter_ref");

    rec_filter.compile_to_lowered_stmt("build/generated_fct_recfilter_ref.txt", {in}, Text);

    return 0;
}

