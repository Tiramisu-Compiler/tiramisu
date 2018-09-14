#include "gaussian.h"
#include <assert.h>
#include <stdint.h>

//#if !__PENCIL__
#include <stdlib.h>
//#endif

#ifdef __PENCIL_HEADER__
static void gaussian( const int rows
                    , const int cols
                    , const int step
                    , const uint8_t src[static const restrict rows][step][3]
                    , const int kernelX_length
                    , const float kernelX[static const restrict kernelX_length]
                    , const int kernelY_length
                    , const float kernelY[static const restrict kernelY_length]
                    , uint8_t conv[static const restrict rows][step][3]
                    , uint8_t temp[rows][step][3]
                    )
#else
static void gaussian( const int rows
                    , const int cols
                    , const int step
                    , const uint8_t *src
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
                    temp[q][w][cc] = prod1;
                }
            }
        }
        for ( int q = 0; q < rows; q++ )
        {
            for ( int w = 0; w < cols; w++ )
            {
                for (int cc = 0; cc < 3; cc++)
                {
                    float prod2 = 0.;
                    for ( int e = 0; e < kernelY_length; e++ )
                    {
                        prod2 += temp[q][w][cc] * kernelY[e];
                    }
                    conv[q][w][cc] = prod2;
                }
            }
        }
#pragma endscop
}

void pencil_gaussian( const int rows
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
    gaussian( rows, cols, step, src
            , kernelX_length, kernelX
            , kernelY_length, kernelY
            , conv
            , temp
            );
}
