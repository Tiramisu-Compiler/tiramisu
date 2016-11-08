#include "Halide.h"

using namespace Halide;


int main(int argc, char* argv[]) {
    ImageParam in{Float(32), 2, "input"};

    Param<float> a0{"a0"};
    Param<float> a1{"a1"};
    Param<float> a2{"a2"};

    Func rec_filter{"rec_filter"};
    Var x, y;
    RDom r(2, in.width()-3, 0, in.height()-1);

    rec_filter(x, y) = in(x, y);
    rec_filter(r.x, r.y) = a0*rec_filter(r.x, r.y)
                         + a1*rec_filter(r.x-1, r.y)
                         + a2*rec_filter(r.x-2, r.y);

    rec_filter.parallel(y);//.vectorize(x, 8);
    rec_filter.update(0).parallel(r.y);

    rec_filter.compile_to_object("build/generated_fct_recfilter_ref.o", {in, a0, a1, a2}, "recfilter_ref");

    rec_filter.compile_to_lowered_stmt("build/generated_fct_recfilter_ref.txt", {in, a0, a1, a2}, Text);

    return 0;
}

