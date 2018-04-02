#include "Halide.h"

using namespace Halide;

int main(int argc, char **argv) {
    ImageParam in(UInt(8), 3, "input");

    Var x("x"), y("y"), c("c"), x1, y1;
    Func f("f"), g("g"), h("h"), k("k");

    f(x, y, c) = cast<uint8_t>(255 - in(x, y, c));
    g(x, y, c) = cast<uint8_t>(2*in(x, y, c));
    h(x, y, c) = f(x, y, c) + g(x, y, c);
    k(x, y, c) = f(x, y, c) - g(x, y, c);

    //g.compute_with(f, y);
    f.gpu_tile(x, y, x1, y1, 16, 16);
    g.gpu_tile(x, y, x1, y1, 16, 16);
    h.gpu_tile(x, y, x1, y1, 16, 16);
    k.gpu_tile(x, y, x1, y1, 16, 16);

    Halide::Target target = Halide::get_host_target();

    Pipeline({f, g, h, k}).compile_to_object("build/generated_fct_fusion_ref.o", {in}, "fusion_ref", target);

    Pipeline({f, g, h, k}).compile_to_lowered_stmt("build/generated_fct_fusion_ref.txt", {in}, Halide::Text, target);

    return 0;
}
