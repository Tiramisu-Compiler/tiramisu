#include "cvtColor.h"

#include <assert.h>
#include <stdint.h>
// #if !__PENCIL__
#include <stdlib.h>
// #endif

#define CV_DESCALE(x, n) (((x) + (1 << ((n)-1))) >> (n))
enum {
    yuv_shift  = 14,
    R2Y        = 4899,
    G2Y        = 9617,
    B2Y        = 1868,
};

#ifdef __NVCC__
static void cvtColor(const int rows,
                     const int cols,
                     const int src_step,
                     const int dst_step,
                     const uint8_t *src,
                     uint8_t *dst) {
#else
static void cvtColor(const int rows,
                     const int cols,
                     const int src_step,
                     const int dst_step,
                     const uint8_t src[rows][src_step][3],
                     uint8_t dst[rows][dst_step]) {
#endif
#pragma scop
    __pencil_assume(rows > 0);
    __pencil_assume(cols > 0);
    __pencil_assume(src_step >= cols);
    __pencil_assume(dst_step >= cols);

    __pencil_kill(dst);
    #pragma pencil independent
    for (int q = 0; q < rows; q++) {
        #pragma pencil independent
        for (int w = 0; w < cols; w++) {
            dst[q][w] = CV_DESCALE((src[q][w][2] * B2Y + src[q][w][1] * G2Y + src[q][w][0] * R2Y), yuv_shift);
        }
    }
    __pencil_kill(src);
#pragma endscop
}

void pencil_cvtColor(const int rows,
                     const int cols,
                     const int src_step,
                     const int dst_step,
                     const uint8_t src[],
                     uint8_t dst[]) {
    cvtColor(rows, cols, src_step, dst_step, src, dst);
}
