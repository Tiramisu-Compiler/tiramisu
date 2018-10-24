#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
void pencil_fusion(const int rows,
                   const int cols,
                   const int step,
                   const uint8_t src[],
                   uint8_t f[],
                   uint8_t g[],
                   uint8_t h[],
                   uint8_t k[]);
#ifdef __cplusplus
}  // extern "C"
#endif
