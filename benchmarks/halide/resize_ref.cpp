#include "Halide.h"

#define NOCLAMP 0

using namespace Halide;

Expr mixf(Expr x, Expr y, Expr a) {
    return x * (1.0f-a) + y * a;
}

int main(int argc, char* argv[]) {
    ImageParam in{UInt(8), 2, "input"};
    Param<int> resampled_rows;
    Param<int> resampled_cols;

    Func resampled{"resampled"};
    Var x("x"), y("y");

    // Translating this algorithm as close as possible

    Expr o_h = in.height();
    Expr o_w = in.width();
    Expr n_h = resampled_rows;
    Expr n_w = resampled_cols;

    Expr n_r = y;
    Expr n_c = x;

    Expr o_r = (n_r + 0.5f) * (o_h) / (n_h) - 0.5f;
    Expr o_c = (n_c + 0.5f) * (o_w) / (n_w) - 0.5f;

    Expr r = o_r - floor(o_r);
    Expr c = o_c - floor(o_c);

#ifdef NOCLAMP
    Expr coord_00_r = cast<int>(floor(o_r));
    Expr coord_00_c = cast<int>(floor(o_c));
#else
    Expr coord_00_r = clamp( cast<int>(floor(o_r)), 0, o_h - 1 );
    Expr coord_00_c = clamp( cast<int>(floor(o_c)), 0, o_w - 1 );
#endif

    Expr coord_01_r = coord_00_r;

#ifdef NOCLAMP
    Expr coord_01_c = coord_00_c + 1;
    Expr coord_10_r = coord_00_r + 1;
#else
    Expr coord_01_c = clamp( coord_00_c + 1, 0, o_w - 1 );
    Expr coord_10_r = clamp( coord_00_r + 1, 0, o_h - 1 );
#endif

    Expr coord_10_c = coord_00_c;

#ifdef NOCLAMP
    Expr coord_11_r = coord_00_r + 1;
    Expr coord_11_c = coord_00_c + 1;
#else
    Expr coord_11_r = clamp( coord_00_r + 1, 0, o_h - 1 );
    Expr coord_11_c = clamp( coord_00_c + 1, 0, o_w - 1 );
#endif

    Expr A00 = in(coord_00_r, coord_00_c);
    Expr A10 = in(coord_10_r, coord_10_c);
    Expr A01 = in(coord_01_r, coord_01_c);
    Expr A11 = in(coord_11_r, coord_11_c);

    resampled(x, y) = mixf( mixf(A00, A10, r), mixf(A01, A11, r), c);

    resampled.parallel(y).vectorize(x, 8);

    resampled.compile_to_object("build/generated_fct_resize_ref.o", {in, resampled_rows, resampled_cols}, "resize_ref");

    resampled.compile_to_lowered_stmt("build/generated_fct_resize_ref.txt", {in, resampled_rows, resampled_cols}, HTML);

    return 0;
}