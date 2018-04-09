#include "Halide.h"

using namespace Halide;

int main(int argc, char **argv) {
    ImageParam rgb(UInt(8), 3);

    Var x("x"), y("y");
    Var x0("x0"), x1("x1"), y0("y0"), y1("y1");
    Func y_part("y_part"), u_part("u_part"), v_part("v_part");

    y_part(x, y) = cast<uint8_t>(((66 * cast<int>(rgb(x, y, 0)) + 129 * cast<int>(rgb(x, y, 1)) +  25 * cast<int>(rgb(x, y, 2)) + 128) % 256) +  16);
    u_part(x, y) = cast<uint8_t>((( -38 * cast<int>(rgb(2*x, 2*y, 0)) -  cast<int>(74 * rgb(2*x, 2*y, 1)) + 112 * cast<int>(rgb(2*x, 2*y, 2) + 128)) % 256) + 128);
    v_part(x, y) = cast<uint8_t>((( 112 * cast<int>(rgb(2*x, 2*y, 0)) -  cast<int>(94 * rgb(2*x, 2*y, 1)) -  18 * cast<int>(rgb(2*x, 2*y, 2) + 128)) % 256) + 128);

    //u_part.compute_with(y_part, y);
    //v_part.compute_with(u_part, y);
    y_part.gpu_tile(x, y, x1, y1, 16, 16);
    u_part.gpu_tile(x, y, x1, y1, 16, 16);
    v_part.gpu_tile(x, y, x1, y1, 16, 16);
    y_part.compute_root();
    u_part.compute_root();
    v_part.compute_root();

    Halide::Target target = Halide::get_host_target();
    target.set_feature(Target::Feature::CUDA, true);

    Pipeline({y_part, u_part, v_part}).compile_to_object("build/generated_fct_rgbyuv420gpu_ref.o", {rgb}, "rgbyuv420gpu_ref", target);

    Pipeline({y_part, u_part, v_part}).compile_to_lowered_stmt("build/generated_fct_rgbyuv420gpu_ref.txt", {rgb}, Halide::Text, target);

    return 0;
}
