#include "Halide.h"

using namespace Halide;

int main(int argc, char **argv) {
    ImageParam in(UInt(8), 3, "input");

    Var x("x"), y("y"), c("c");
    Func f("f"), g("g");

    f(x, y, c) = cast<uint8_t>(255 - in(x, y, c));
    g(x, y, c) = cast<uint8_t>(2*in(x, y, c));

    //g.compute_with(f, y);

    Halide::Target target = Halide::get_host_target();

    Pipeline({f, g}).compile_to_object("build/generated_fct_fusion_ref.o", {in}, "fusion_ref", target);

    Pipeline({f, g}).compile_to_lowered_stmt("build/generated_fct_fusion_ref.txt", {in}, Halide::Text, target);

    return 0;
}
