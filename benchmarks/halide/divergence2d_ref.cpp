#include "Halide.h"
using namespace Halide;

int main(int argc, char **argv) {
    ImageParam in(Float(32), 2, "input");

    float alpha = 0.3;
    float beta = 0.4;

    Func divergence2d("divergence2d");
    Var x("x"), y("y");

    RDom r(1, in.width()-2, 1, in.height()-2);
    divergence2d(x, y) = 0.0f;
    divergence2d(r.x, r.y) = alpha * (in(r.x+1, r.y) + in(r.x-1, r.y)) +
                             beta  * (in(r.x, r.y+1) + in(r.x, r.y-1));

    divergence2d.parallel(y).vectorize(x, 8);
    divergence2d.update().parallel(r.y).vectorize(r.x, 8);

    Halide::Target target = Halide::get_host_target();

    divergence2d.compile_to_object("build/generated_fct_divergence2d_ref.o",
                                   {in},
                                   "divergence2d_ref",
                                   target);

    divergence2d.compile_to_lowered_stmt("build/generated_fct_divergence2d_ref.txt",
                                         {in},
                                         Text,
                                         target);

    return 0;
}
