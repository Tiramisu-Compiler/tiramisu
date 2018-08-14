#include "gaussian.h"
#include <pencil.h>
#include <assert.h>

//#if !__PENCIL__
#include <stdlib.h>
//#endif

static void gaussian( const int rows
                    , const int cols
                    , const int step
                    , const uint8_t src[static const restrict rows][step]
                    , const int kernelX_length
                    , const float kernelX[static const restrict kernelX_length]
                    , const int kernelY_length
                    , const float kernelY[static const restrict kernelY_length]
                    , uint8_t conv[static const restrict rows][step]
		    , uint8_t temp[rows][step]
                    )
{
#pragma scop
    __pencil_assume(rows         >  0);
    __pencil_assume(cols         >  0);
    __pencil_assume(step         >= cols);
    __pencil_assume(kernelX_length >  0);
    __pencil_assume(kernelX_length <= 64);
    __pencil_assume(kernelY_length >  0);
    __pencil_assume(kernelY_length <= 64);
    
    __pencil_kill(conv);
    {
        #pragma pencil independent
        for ( int q = 0; q < rows; q++ )
        {
            #pragma pencil independent
            for ( int w = 0; w < cols; w++ )
            {
                float prod1 = 0.;
                #pragma pencil independent reduction (+: prod1);
                for ( int r = 0; r < kernelX_length; r++ )
                {
                    int row1 = q;
                    int col1 = clamp(w + r - kernelX_length / 2, 0, cols-1);
                    prod1 += src[row1][col1] * kernelX[r];
                }
                temp[q][w] = prod1;
            }
        }
        #pragma pencil independent
        for ( int q = 0; q < rows; q++ )
        {
            #pragma pencil independent
            for ( int w = 0; w < cols; w++ )
            {
                float prod2 = 0.;
                #pragma pencil independent reduction (+: prod2);
                for ( int e = 0; e < kernelY_length; e++ )
                {
                    int row2 = clamp(q + e - kernelY_length / 2, 0, rows-1);
                    int col2 = w;
                    prod2 += temp[row2][col2] * kernelY[e];
                }
                conv[q][w] = prod2;
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
    gaussian( rows, cols, step, (const uint8_t(*)[step])src
            , kernelX_length, kernelX
            , kernelY_length, kernelY
            , (uint8_t(*)[step])conv
	    , temp
            );
}
