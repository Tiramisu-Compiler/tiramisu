#ifndef BENCH_CVT_COLOR_H_
#define BENCH_CVT_COLOR_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
void pencil_cvtColor(const int rows,
                     const int cols,
                     const int src_step,
                     const int dst_step,
                     const uint8_t src[],
                     uint8_t dst[]);
#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // BENCH_CVT_COLOR_H_
