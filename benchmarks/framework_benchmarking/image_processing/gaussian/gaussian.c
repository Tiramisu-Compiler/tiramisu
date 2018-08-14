#include "gaussian.h"
#include <pencil.h>
#include <assert.h>

//#if !__PENCIL__
#include <stdlib.h>
//#endif

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
		for (int cc = 0; cc < 3; cc++)
		{
		    float prod1 = 0.;
		    #pragma pencil independent reduction (+: prod1);
		    for ( int r = 0; r < kernelX_length; r++ )
		    {
			int row1 = q;
			int col1 = clamp(w + r - kernelX_length / 2, 0, cols-1);
			prod1 += src[row1][col1][cc] * kernelX[r];
		    }
		    temp[q][w][cc] = prod1;
		}
            }
        }
        #pragma pencil independent
        for ( int q = 0; q < rows; q++ )
        {
            #pragma pencil independent
            for ( int w = 0; w < cols; w++ )
            {
		for (int cc = 0; cc < 3; cc++)
		{
		    float prod2 = 0.;
		    #pragma pencil independent reduction (+: prod2);
		    for ( int e = 0; e < kernelY_length; e++ )
		    {
			int row2 = clamp(q + e - kernelY_length / 2, 0, rows-1);
			int col2 = w;
			prod2 += temp[row2][col2][cc] * kernelY[e];
		    }
		    conv[q][w][cc] = prod2;
		}
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
