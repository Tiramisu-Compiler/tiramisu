#include "gaussian.h"

#include <stdint.h>
#if !__PENCIL__
#include <stdlib.h>
#endif

/*
 * Note that this functions is not same as PPCG benchmark.
 * It is meant to be consistent with Halide/Tiramisu benchmarks for gaussian.
 * TODO: make sure data layout is the same
 */
#ifdef __NVCC__
static void gaussian(const int rows,
                     const int cols,
                     const int step,
                     uint8_t *src,
                     float *kernelX,
                     float *kernelY,
                     uint8_t *temp,
                     uint8_t *conv) {
#else
static void gaussian(const int rows,
                     const int cols,
                     const int step,
                     uint8_t src[static const restrict rows][step][3],
                     float kernelX[5],
                     float kernelY[5],
                     uint8_t temp[static const restrict rows][step][3],
                     uint8_t conv[static const restrict rows][step][3]) {
#endif
#pragma scop
    __pencil_assume(rows > 0);
    __pencil_assume(cols > 0);
    __pencil_assume(step >= cols);
    __pencil_kill(temp);
    __pencil_kill(conv);
    #pragma pencil independent
    for (int q = 0; q < rows - 5; q++) {
        #pragma pencil independent
        for (int w = 0; w < cols - 5; w++) {
            #pragma pencil independent
            for (int cc = 0; cc < 3; cc++) {
                float prod1 = 0.;
                #pragma pencil independent reduction (+: prod1);
                for (int r = 0; r < 5; r++) {
                    prod1 += src[q + r][w][cc] * kernelX[r];
                }
                temp[q][w][cc] = prod1;
            }
        }
    }
    #pragma pencil independent
    for (int q = 0; q < rows - 5; q++) {
        #pragma pencil independent
        for (int w = 0; w < cols - 5; w++) {
            #pragma pencil independent
            for (int cc = 0; cc < 3; cc++) {
                float prod2 = 0.;
                #pragma pencil independent reduction (+: prod2);
                for (int e = 0; e < 5; e++) {
                    prod2 += temp[q][w + e][cc] * kernelY[e];
                }
                conv[q][w][cc] = prod2;
            }
        }
    }
    __pencil_kill(temp);
    __pencil_kill(src);
#pragma endscop
}

void pencil_gaussian(const int rows,
                     const int cols,
                     const int step,
                     uint8_t src[],
                     float kernelX[],
                     float kernelY[],
                     uint8_t temp[],
                     uint8_t conv[]) {
    gaussian(rows, cols, step, src, kernelX, kernelY, temp, conv);
}
