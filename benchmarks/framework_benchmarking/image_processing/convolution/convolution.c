#include "convolution.h"
#include <assert.h>
#include <stdint.h>

//#if !__PENCIL__
#include <stdlib.h>
//#endif

#ifdef __PENCIL_HEADER__
static void convolution( const int rows
                    , const int cols
                    , const int step
                    , const unsigned char src[rows][step][3]
                    , const int kernelX_length
                    , float kernelX[kernelX_length]
                    , const int kernelY_length
                    , float kernelY[kernelY_length]
                    , uint8_t conv[rows][step][3]
                    , uint8_t temp[rows][step][3]
                    )
#else
static void convolution( const int rows
                    , const int cols
                    , const int step
                    , const unsigned char *src
                    , const int kernelX_length
                    , float *kernelX
                    , const int kernelY_length
                    , float *kernelY
                    , uint8_t *conv
                    , uint8_t *temp
                    )
#endif
{
#pragma scop
    {
        for ( int q = 0; q < rows; q++ )
        {
            for ( int w = 0; w < cols; w++ )
            {
                for (int cc = 0; cc < 3; cc++)
                {
                    float prod1 = 0.;
                    for ( int r = 0; r < kernelX_length; r++ )
                    {
                        prod1 += src[q][w][cc] * kernelX[r];
                    }
                    conv[q][w][cc] = prod1;
                }
            }
        }
    }
#pragma endscop
}

void pencil_convolution( const int rows
                    , const int cols
                    , const int step
                    , const uint8_t src[]
                    , const int kernelX_length
                    , float kernelX[]
                    , const int kernelY_length
                    , float kernelY[]
                    , uint8_t conv[]
                    , uint8_t temp[]
                    )
{
    convolution( rows, cols, step, src
            , kernelX_length, kernelX
            , kernelY_length, kernelY
            , conv
            , temp
            );
}
