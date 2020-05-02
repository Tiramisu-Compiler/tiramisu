#ifndef __CONV_CONF_HEADER_
#define __CONV_CONF_HEADER_

#define LARGE_DATA_SET	0
#define MEDIUM_DATA_SET	0
#define SMALL_DATA_SET	1

#if LARGE_DATA_SET
    #define BATCH_SIZE 100
#elif MEDIUM_DATA_SET
    #define BATCH_SIZE 32
#elif SMALL_DATA_SET
    #define BATCH_SIZE 8
#endif

// Width and height of an input tensor
#define INP_X 1024
#define INP_Y 1024

// Number of channels
#define CHANNELS 3


// Convolution kernel size
#define K_X 3 
#define K_Y 3

// Number of kernels
#define NB_K 2




template<typename T>
inline void init_4D_buffer(Halide::Buffer<T> &buf, T val)
{

    for (int z = 0; z < buf.dim(3).extent(); z++)
    {
        for (int y = 0; y < buf.dim(2).extent(); y++)
        {
            for (int x = 0; x < buf.dim(1).extent(); x++)
            {
               for (int u = 0; u < buf.dim(0).extent(); u++)
               {
                    buf(u, x, y, z) = val;
               }
            }
        }
    }
}

#endif