#include "tiramisu/tiramisu.h"

using namespace tiramisu;

#define mixf(x, y, a) (cast(p_float32, x) * (expr((float) 1) - cast(p_float32, a)) + cast(p_float32, y) * cast(p_float32, a))

int main(int argc, char* argv[])
{
    tiramisu::init("warp_affine_tiramisu");

    Input SIZES("SIZES", {2}, p_int32);

    constant N0("N0", SIZES(0));
    constant N1("N1", SIZES(1));

    Input in("in", {N0, N1}, p_uint8);

    var x("x", 0, N1), y("y", 0, N0);

    expr a00 = expr((float) 0.1);
    expr a01 = expr((float) 0.1);
    expr a10 = expr((float) 0.1);
    expr a11 = expr((float) 0.1);
    expr b00 = expr((float) 0.1);
    expr b10 = expr((float) 0.1);

    expr o_r = a11*cast(p_float32, y) + a10*cast(p_float32, x) + b00;
    expr o_c = a01*cast(p_float32, y) + a00*cast(p_float32, x) + b10;

    expr r = o_r - floor(o_r);
    expr c = o_c - floor(o_c);

    expr coord_00_r = cast(p_int32, floor(o_r));
    expr coord_00_c = cast(p_int32, floor(o_c));
    expr coord_01_r = cast(p_int32, coord_00_r);
    expr coord_01_c = cast(p_int32, coord_00_c + 1);
    expr coord_10_r = cast(p_int32, coord_00_r + 1);
    expr coord_10_c = cast(p_int32, coord_00_c);
    expr coord_11_r = cast(p_int32, coord_00_r + 1);
    expr coord_11_c = cast(p_int32, coord_00_c + 1);

    coord_00_r = clamp(coord_00_r, 0, N0);
    coord_00_c = clamp(coord_00_c, 0, N1);
    coord_01_r = clamp(coord_01_r, 0, N0);
    coord_01_c = clamp(coord_01_c, 0, N1);
    coord_10_r = clamp(coord_10_r, 0, N0);
    coord_10_c = clamp(coord_10_c, 0, N1);
    coord_11_r = clamp(coord_11_r, 0, N0);
    coord_11_c = clamp(coord_11_c, 0, N1);

    expr A00 = in(coord_00_c, coord_00_r);
    expr A10 = in(coord_10_c, coord_10_r);
    expr A01 = in(coord_01_c, coord_01_r);
    expr A11 = in(coord_11_c, coord_11_r);

    expr e = cast(p_float32, mixf(mixf(A00, A10, r), mixf(A01, A11, r), c));

    computation affine({y, x}, e);

    affine.parallelize(y);
    affine.vectorize(x, 16);

    buffer  b_input("b_input",  {N0, N1}, p_uint8, a_input);
    buffer b_SIZES("b_SIZES", {2}, p_int32, a_input);
    buffer b_affine("b_affine", {N0, N1}, p_float32, a_output);
    in.store_in(&b_input);
    SIZES.store_in(&b_SIZES);
    affine.store_in(&b_affine);

    tiramisu::codegen({&b_SIZES, &b_input, &b_affine}, "build/generated_fct_warp_affine.o");

    global::get_implicit_function()->dump_halide_stmt();

    return 0;
}
