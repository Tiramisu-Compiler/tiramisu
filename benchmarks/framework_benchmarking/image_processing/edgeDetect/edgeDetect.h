#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
void pencil_edge(const int rows,
                 const int cols,
                 const int step,
                 const uint8_t src[],
                 uint8_t out[],
                 uint8_t temp[]);
#ifdef __cplusplus
}  // extern "C"
#endif
