#include "Halide.h"
using namespace Halide;

constexpr auto block_size = 16;

int main(int argc, char **argv) {
    ImageParam in(Float(32), 2, "input");

    float alpha = 0.3;
    float beta = 0.4;

    Func heat2d("heat2dgpu");
    Var x("x"), y("y"), x0, y0, x1, y1, rx0, ry0, rx1, ry1;

    RDom r(1, in.width()-2, 1, in.height()-2);
    heat2d(x, y) = 0.0f;
    heat2d(r.x, r.y) = alpha * in(r.x, r.y) +
                       beta * (in(r.x+1, r.y) + in(r.x-1, r.y) + in(r.x, r.y+1) + in(r.x, r.y-1));

    heat2d.gpu_tile(x, y, x0, y0, x1, y1, block_size, block_size);
    heat2d.update().gpu_tile(r.x, r.y, block_size, block_size);

    Halide::Target target = Halide::get_host_target();
    target.set_feature(Target::Feature::CUDA, true);

    heat2d.compile_to_object("build/generated_fct_heat2dgpu_ref.o",
                             {in},
                             "heat2d_ref",
                             target);

    heat2d.compile_to_lowered_stmt("build/generated_fct_heat2dgpu_ref.txt",
                                   {in},
                                   Text,
                                   target);
    return 0;
}
