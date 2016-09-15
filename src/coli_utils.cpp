#include "Halide.h"
#include "coli/utils.h"

void print_array_1D(buffer_t *buf, int N)
{
    int i;

    for (i=0; i<N; i++)
    {
       printf("%u, ", buf->host[i]);
    }
    printf("\n");
}

void init_array_1D(buffer_t *buf, int N, uint8_t val)
{
    int i;

    for (i=0; i<N; i++)
        buf->host[i] = val;
}

void print_array_2D(buffer_t buf, int N, int M)
{
    int i,j;

    for (i=0; i<N; i++)
    {
        for (j=0; j<M; j++)
        {
            printf("%u, ", buf.host[i*M+j]);
        }
        printf("\n");
    }
    printf("\n");
}

void init_array_val_2D(buffer_t *buf, int N, int M, uint8_t val)
{
    int i, j;

    for (i=0; i<N; i++)
        for (j=0; j<M; j++)
            buf->host[i*M+j] = val;
}

void copy_array_2D(uint8_t* buf, int N, int M, uint8_t* array)
{
    int i, j;

    for (i=0; i<N; i++)
        for (j=0; j<M; j++)
            buf[i*M+j] = array[i*M+j];
}
