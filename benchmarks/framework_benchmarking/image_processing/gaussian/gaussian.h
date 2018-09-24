#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
void pencil_gaussian(const int rows,
                     const int cols,
                     const int step,
                     uint8_t src[],
                     float kernelX[],
                     float kernelY[],
                     uint8_t temp[],
                     uint8_t conv[]);
#ifdef __cplusplus
}  // extern "C"
#endif
