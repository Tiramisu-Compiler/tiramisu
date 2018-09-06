#include "fusion.h"
#include <pencil.h>
#include <assert.h>

//#if !__PENCIL__
#include <stdlib.h>
//#endif

static void fusion( const int rows
                    , const int cols
                    , const int step
                    , const unsigned char src[rows][step][3]
                    , const int kernelX_length
                    , const float kernelX[kernelX_length]
                    , const int kernelY_length
                    , const float kernelY[kernelY_length]
                    , uint8_t f[rows][step][3]
		    , uint8_t g[rows][step][3]
                    , uint8_t h[rows][step][3]
		    , uint8_t k[rows][step][3]
                    )
{
#pragma scop
        for ( int q = 0; q < rows; q++ )
            for ( int w = 0; w < cols; w++ )
		for (int cc = 0; cc < 3; cc++)
		{
			f[q][w][cc] = 255 - src[q][w][cc];
			g[q][w][cc] =   2 * src[q][w][cc];
			h[q][w][cc] = f[q][w][cc] + g[q][w][cc];
			k[q][w][cc] = f[q][w][cc] - g[q][w][cc];
		}
#pragma endscop
}

void pencil_fusion( const int rows
                    , const int cols
                    , const int step
                    , const uint8_t src[]
                    , const int kernelX_length
                    , const float kernelX[]
                    , const int kernelY_length
                    , const float kernelY[]
                    , uint8_t f[]
		    , uint8_t g[]
                    , uint8_t h[]
		    , uint8_t k[]
                    )
{
    fusion( rows, cols, step, (const uint8_t(*)[step])src
            , kernelX_length, kernelX
            , kernelY_length, kernelY
            , (uint8_t(*)[step])f
            , (uint8_t(*)[step])g
            , (uint8_t(*)[step])h
            , (uint8_t(*)[step])k
            );
}
