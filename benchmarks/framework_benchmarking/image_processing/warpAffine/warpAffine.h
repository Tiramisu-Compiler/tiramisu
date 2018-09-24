#ifndef WARPAFFINE_PENCIL_H
#define WARPAFFINE_PENCIL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void pencil_affine_linear(const int src_rows, const int src_cols, const int src_step, const uint8_t src[],
                          const int dst_rows, const int dst_cols, const int dst_step,         float dst[],
                          const float a00, const float a01, const float a10, const float a11, const float b00, const float b10);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif
