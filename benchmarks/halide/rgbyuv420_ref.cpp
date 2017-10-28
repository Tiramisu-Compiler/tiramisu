#include "Halide.h"

using namespace Halide;

int main(int argc, char **argv) {
    ImageParam rgb(UInt(8), 3);

    Var x("x"), y("y");
    Func y_part("y_part"), u_part("u_part"), v_part("v_part");

    y_part(x, y) = cast<uint8_t>((((uint8_t)66 * rgb(x, y, 0) + (uint8_t)129 * rgb(x, y, 1) +  (uint8_t)25 * rgb(x, y, 2) + (uint8_t)128) >> (uint8_t)8) +  (uint8_t)16);
    u_part(x, y) = cast<uint8_t>((( (uint8_t)38 * rgb(2*x, 2*y, 0) -  (uint8_t)74 * rgb(2*x, 2*y, 1) + (uint8_t)112 * rgb(2*x, 2*y, 2) + (uint8_t)128) >> (uint8_t)8) + (uint8_t)128);
    v_part(x, y) = cast<uint8_t>((( (uint8_t)112 * rgb(2*x, 2*y, 0) -  (uint8_t)94 * rgb(2*x, 2*y, 1) -  (uint8_t)18 * rgb(2*x, 2*y, 2) + (uint8_t)128) >> (uint8_t)8) + (uint8_t)128);

    //u_part.compute_with(y_part, y);
    //v_part.compute_with(u_part, y);
    y_part.parallel(y).vectorize(x, 8);
    u_part.parallel(y).vectorize(x, 8);
    v_part.parallel(y).vectorize(x, 8);

    Pipeline({y_part, u_part, v_part}).compile_to_object("build/generated_fct_rgbyuv420_ref.o", {rgb}, "rgbyuv420_ref");

    Pipeline({y_part, u_part, v_part}).compile_to_lowered_stmt("build/generated_fct_rgbyuv420_ref.txt", {rgb}, Halide::Text);

    return 0;
}
