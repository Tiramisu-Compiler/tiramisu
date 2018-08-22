#include "edgeDetect.h"
#include <pencil.h>
#include <assert.h>

//#if !__PENCIL__
#include <stdlib.h>
//#endif

static void     edge( const int rows
                    , const int cols
                    , const int step
                    , const uint8_t src[static const restrict rows][step][3]
                    , uint8_t out[static const restrict rows][step][3]
		    , uint8_t temp[rows][step][3]
                    )
{
#pragma scop
        #pragma pencil independent
        for ( int i = 0; i < rows-2; i++ )
        {
            #pragma pencil independent
            for ( int j = 0; j < cols-2; j++ )
            {
		for (int c = 0; c < 3; c++)
		{
		    temp[i][j][c] = (src[i][j][c]   + src[i][j+1][c]   + src[i][j+2][c]+
				     src[i+1][j][c]                    + src[i+1][j+2][c]+
			             src[i+2][j][c] + src[i+2][j+1][c] + src[i+2][j+2][c])/((uint8_t) 8);
		}
            }
        }
        #pragma pencil independent
        for ( int i = 0; i < rows-2; i++ )
        {
            #pragma pencil independent
            for ( int j = 0; j < cols-2; j++ )
            {
		for (int c = 0; c < 3; c++)
		{
		    out[i][j][c] = (temp[i+1][j+1][c]-temp[i+2][j][c]) + (temp[i+2][j+1][c]-temp[i+1][j][c]);
		}
            }
        }
#pragma endscop
}

void pencil_edge( const int rows
                    , const int cols
                    , const int step
                    , const uint8_t src[]
                    , uint8_t out[]
		    , uint8_t temp[]
                    )
{
    edge    ( rows, cols, step, (const uint8_t(*)[step])src
            , (uint8_t(*)[step])out
	    , temp
            );
}
