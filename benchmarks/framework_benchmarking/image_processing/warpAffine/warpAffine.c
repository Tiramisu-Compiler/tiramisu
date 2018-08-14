#include "warpAffine.h"
//#include <pencil.h>

float mixf(float x, float y, float a) { return x + (x-y) * a; }
int clamp(int x, int minval, int maxval) { if (x < minval) return minval; if (x > maxval) return maxval; return x; }

static void affine( const int src_rows, const int src_cols, const int src_step, const float src[static const restrict src_rows][src_step][3]
                  , const int dst_rows, const int dst_cols, const int dst_step,       float dst[static const restrict dst_rows][dst_step][3]
                  , const float a00, const float a01, const float a10, const float a11, const float b00, const float b10
                  )
{
#pragma scop
    __pencil_assume(src_rows >  0);
    __pencil_assume(src_cols >  0);
    __pencil_assume(src_step >= src_cols);
    __pencil_assume(dst_rows >  0);
    __pencil_assume(dst_cols >  0);
    __pencil_assume(dst_step >= dst_cols);

    __pencil_kill(dst);

    #pragma pencil independent
    for ( int n_r=0; n_r<dst_rows; n_r++ )
    {
        #pragma pencil independent
        for ( int n_c=0; n_c<dst_cols; n_c++ )
        {
	    for (int cc = 0; cc < 3; cc++)
	    {
		float o_r = a11 * n_r + a10 * n_c + b00;
		float o_c = a01 * n_r + a00 * n_c + b10;

		float r = o_r - floorf(o_r);
		float c = o_c - floorf(o_c);

		int coord_00_r = floorf(o_r);
		int coord_00_c = floorf(o_c);
		int coord_01_r = coord_00_r;
		int coord_01_c = coord_00_c + 1;
		int coord_10_r = coord_00_r + 1;
		int coord_10_c = coord_00_c;
		int coord_11_r = coord_00_r + 1;
		int coord_11_c = coord_00_c + 1;

		coord_00_r = clamp(coord_00_r, 0, src_rows);
		coord_00_c = clamp(coord_00_c, 0, src_cols);
		coord_01_r = clamp(coord_01_r, 0, src_rows);
		coord_01_c = clamp(coord_01_c, 0, src_cols);
		coord_10_r = clamp(coord_10_r, 0, src_rows);
		coord_10_c = clamp(coord_10_c, 0, src_cols);
		coord_11_r = clamp(coord_11_r, 0, src_rows);
		coord_11_c = clamp(coord_11_c, 0, src_cols);

		float A00 = src[coord_00_r][coord_00_c][cc];
		float A10 = src[coord_10_r][coord_10_c][cc];
		float A01 = src[coord_01_r][coord_01_c][cc];
		float A11 = src[coord_11_r][coord_11_c][cc];

		dst[n_r][n_c][cc] = mixf( mixf(A00, A10, r), mixf(A01, A11, r), c);
	    }
        }
    }
    __pencil_kill(src);
#pragma endscop
}

void pencil_affine_linear( const int src_rows, const int src_cols, const int src_step, const float src[]
                         , const int dst_rows, const int dst_cols, const int dst_step,       float dst[]
                         , const float a00, const float a01, const float a10, const float a11, const float b00, const float b10
                         )
{
    affine( src_rows, src_cols, src_step, (const float(*)[src_step][3])src
          , dst_rows, dst_cols, dst_step, (      float(*)[dst_step][3])dst
          , a00, a01, a10, a11, b00, b10
          );
}
