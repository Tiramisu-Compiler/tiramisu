#include "cvtColor.h"
#include <assert.h>
#include <stdint.h>

//#if !__PENCIL__
#include <stdlib.h>
//#endif

#define CV_DESCALE(x,n) (((x) + (1 << ((n)-1))) >> (n))
enum
{
    yuv_shift  = 14,
    R2Y        = 4899,
    G2Y        = 9617,
    B2Y        = 1868,
};

#ifdef __PENCIL_HEADER__
static void cvtColor( const int rows
                    , const int cols
                    , const int step
                    , const unsigned char src[rows][step][3]
                    , const int kernelX_length
                    , const float kernelX[kernelX_length]
                    , const int kernelY_length
                    , const float kernelY[kernelY_length]
                    , uint8_t conv[rows][step]
                    , uint8_t temp[rows][step][3]
                    )
#else
static void cvtColor( const int rows
                    , const int cols
                    , const int step
                    , const unsigned char *src
                    , const int kernelX_length
                    , const float *kernelX
                    , const int kernelY_length
                    , const float *kernelY
                    , uint8_t *conv
                    , uint8_t *temp
                    )
#endif
{
#pragma scop
        for ( int q = 0; q < rows; q++ )
            for ( int w = 0; w < cols; w++ )
                conv[q][w] = CV_DESCALE( (src[q][w][2] * B2Y + src[q][w][1] * G2Y + src[q][w][0] * R2Y ), yuv_shift );
#pragma endscop
}

void pencil_cvtColor( const int rows
                    , const int cols
                    , const int step
                    , const uint8_t src[]
                    , const int kernelX_length
                    , const float kernelX[]
                    , const int kernelY_length
                    , const float kernelY[]
                    , uint8_t conv[]
		    , uint8_t temp[]
                    )
{
    cvtColor( rows, cols, step, src
            , kernelX_length, kernelX
            , kernelY_length, kernelY
            , conv
	    , temp
            );
}
