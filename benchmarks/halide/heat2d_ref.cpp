#include "Halide.h"
using namespace Halide;

int main(int argc, char **argv) {
    ImageParam in(Float(32), 2, "input");

    float alpha = 0.3;
    float beta = 0.4;

    Func heat2d("heat2d");
    Var x("x"), y("y");

    RDom r(1, in.width()-2, 1, in.height()-2);
    heat2d(x, y) = 0.0f;
    heat2d(r.x, r.y) = alpha * in(r.x, r.y) +
                       beta * (in(r.x+1, r.y) + in(r.x-1, r.y) + in(r.x, r.y+1) + in(r.x, r.y-1));

    heat2d.parallel(y).vectorize(x, 8, Halide::TailStrategy::GuardWithIf);
    heat2d.update().parallel(r.y).vectorize(r.x, 8, Halide::TailStrategy::GuardWithIf);

    Halide::Target target = Halide::get_host_target();

    heat2d.compile_to_object("build/generated_fct_heat2d_ref.o",
                             {in},
                             "heat2d_ref",
                             target);

    heat2d.compile_to_lowered_stmt("build/generated_fct_heat2d_ref.txt",
                                   {in},
                                   Text,
                                   target);
    return 0;
}
