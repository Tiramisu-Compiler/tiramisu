#include "Halide.h"

using namespace Halide;

int main(int argc, char **argv) {
    const int f_size = 1024;
    const int g_size = 1024;
    Image<int> f_im(f_size, f_size), g_im(g_size, g_size);

    Var x("x"), y("y");
    Func f("f"), g("g"), input("input");

    input(x, y) = x + y + 1;
    f(x, y) = 100 - input(x, y);
    g(x, y) = x + input(x, y);

    //input.compute_at(f, y);
    //g.compute_with(f, y);

    Halide::Target target = Halide::get_host_target();
    Pipeline({f, g}).compile_to_object("build/generated_fct_fusion_ref.o",
                                       {},
                                       "fusion_ref",
                                       target);

    Pipeline({f, g}).compile_to_lowered_stmt("build/generated_fct_fusion_ref.txt",
                                             {},
                                             Halide::Text,
                                             target);

    return 0;
}
