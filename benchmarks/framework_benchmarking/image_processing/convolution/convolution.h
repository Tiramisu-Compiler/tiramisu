#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
void pencil_convolution(const int rows,
                        const int cols,
                        const int step,
                        const uint8_t src[],
                        float kernel[],
                        uint8_t conv[]);
#ifdef __cplusplus
}  // extern "C"
#endif
