#include "convolution.h"

#include <stdint.h>

#ifdef __NVCC__
static void convolution(const int rows,
                        const int cols,
                        const int step,
                        const uint8_t *src,
                        float *kernel,
                        uint8_t *conv) {
#else
static void convolution(const int rows,
                        const int cols,
                        const int step,
                        const uint8_t src[rows][step][3],
                        float kernel[3][3],
                        uint8_t conv[rows][step][3]) {
#endif
#pragma scop
    __pencil_kill(conv);
    for (int q = 0; q < rows - 2; q++) {
        for (int w = 0; w < cols - 2; w++) {
            for (int cc = 0; cc < 3; cc++) {
                float prod = 0.;
                for (int kq = 0; kq < 3; kq++) {
                    for (int kw = 0; kw < 3; kw++) {
                        prod += src[q + kq][w + kw][cc] * kernel[kq][kw];
                    }
                }
                conv[q][w][cc] = prod;
            }
        }
    }
#pragma endscop
}

void pencil_convolution(const int rows,
                        const int cols,
                        const int step,
                        const uint8_t src[],
                        float kernel[],
                        uint8_t conv[]) {
    convolution(rows, cols, step, src, kernel, conv);
}
