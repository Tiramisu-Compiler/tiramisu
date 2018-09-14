#include "fusion.h"

#include <stdint.h>

#ifdef __NVCC__
static void fusion(const int rows,
                   const int cols,
                   const int step,
                   const uint8_t *src,
                   uint8_t *f,
                   uint8_t *g,
                   uint8_t *h,
                   uint8_t *k) {
#else
static void fusion(const int rows,
                   const int cols,
                   const int step,
                   const uint8_t src[rows][step][3],
                   uint8_t f[rows][step][3],
                   uint8_t g[rows][step][3],
                   uint8_t h[rows][step][3],
                   uint8_t k[rows][step][3]) {
#endif
#pragma scop
    __pencil_kill(f);
    __pencil_kill(g);
    __pencil_kill(h);
    __pencil_kill(k);
    for (int q = 0; q < rows; q++) {
        for (int w = 0; w < cols; w++) {
            for (int cc = 0; cc < 3; cc++) {
                f[q][w][cc] = 255 - src[q][w][cc];
                g[q][w][cc] =   2 * src[q][w][cc];
                h[q][w][cc] = f[q][w][cc] + g[q][w][cc];
                k[q][w][cc] = f[q][w][cc] - g[q][w][cc];
            }
        }
    }
    __pencil_kill(src);
#pragma endscop
}

void pencil_fusion(const int rows,
                   const int cols,
                   const int step,
                   const uint8_t src[],
                   uint8_t f[],
                   uint8_t g[],
                   uint8_t h[],
                   uint8_t k[]) {
    fusion(rows, cols, step, src, f, g, h, k);
}
