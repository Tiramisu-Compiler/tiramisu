#include "Halide.h"

#define NOCLAMP

using namespace Halide;

Expr mixf(Expr x, Expr y, Expr a) {
    return x * (1.0f-a) + y * a;
}

int main(int argc, char* argv[]) {
    ImageParam in{UInt(8), 2, "input"};
    Param<float> a00;
    Param<float> a01;
    Param<float> a10;
    Param<float> a11;
    Param<float> b00;
    Param<float> b10;

    Func affine{"affine"};
    Var x, y;

    // Translating this algorithm as close as possible
    Expr o_r = a11 * y + a10 * x + b00;
    Expr o_c = a01 * y + a00 * x + b10;

    Expr r = o_r - floor(o_r);
    Expr c = o_c - floor(o_c);

    Expr coord_00_r = cast<int>(floor(o_r));
    Expr coord_00_c = cast<int>(floor(o_c));
    Expr coord_01_r = coord_00_r;
    Expr coord_01_c = coord_00_c + 1;
    Expr coord_10_r = coord_00_r + 1;
    Expr coord_10_c = coord_00_c;
    Expr coord_11_r = coord_00_r + 1;
    Expr coord_11_c = coord_00_c + 1;

    Expr src_rows = in.height();
    Expr src_cols = in.width();

#ifdef NOCLAMP
    coord_00_r = clamp(coord_00_r, 0, src_rows);
    coord_00_c = clamp(coord_00_c, 0, src_cols);
    coord_01_r = clamp(coord_01_r, 0, src_rows);
    coord_01_c = clamp(coord_01_c, 0, src_cols);
    coord_10_r = clamp(coord_10_r, 0, src_rows);
    coord_10_c = clamp(coord_10_c, 0, src_cols);
    coord_11_r = clamp(coord_11_r, 0, src_rows);
    coord_11_c = clamp(coord_11_c, 0, src_cols);
#else
    coord_00_r = coord_00_r;
    coord_00_c = coord_00_c;
    coord_01_r = coord_01_r;
    coord_01_c = coord_01_c;
    coord_10_r = coord_10_r;
    coord_10_c = coord_10_c;
    coord_11_r = coord_11_r;
    coord_11_c = coord_11_c;
#endif

    Expr A00 = in(coord_00_r, coord_00_c);
    Expr A10 = in(coord_10_r, coord_10_c);
    Expr A01 = in(coord_01_r, coord_01_c);
    Expr A11 = in(coord_11_r, coord_11_c);

    affine(x, y) = mixf( mixf(A00, A10, r), mixf(A01, A11, r), c);

    affine.parallel(y).vectorize(x, 8);

    affine.compile_to_object("build/generated_fct_warp_affine_ref.o", {in, a00, a01, a10, a11, b00, b10}, "warp_affine_ref");

    affine.compile_to_lowered_stmt("build/generated_fct_warp_affine_ref.txt", {in, a00, a01, a10, a11, b00, b10}, HTML);

    return 0;
}