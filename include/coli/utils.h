#ifndef _COLI_UTILS
#define _COLI_UTILS

#ifndef HALIDE_ATTRIBUTE_ALIGN
    #ifdef _MSC_VER
        #define HALIDE_ATTRIBUTE_ALIGN(x) __declspec(align(x))
    #else
        #define HALIDE_ATTRIBUTE_ALIGN(x) __attribute__((aligned(x)))
    #endif
#endif
#ifndef BUFFER_T_DEFINED
#define BUFFER_T_DEFINED
#include <stdbool.h>
#include <stdint.h>
typedef struct buffer_t {
    uint64_t dev;
    uint8_t* host;
    int32_t extent[4];
    int32_t stride[4];
    int32_t min[4];
    int32_t elem_size;
    HALIDE_ATTRIBUTE_ALIGN(1) bool host_dirty;
    HALIDE_ATTRIBUTE_ALIGN(1) bool dev_dirty;
    HALIDE_ATTRIBUTE_ALIGN(1) uint8_t _padding[10 - sizeof(void *)];
} buffer_t;
#endif
struct halide_filter_metadata_t;
#ifndef HALIDE_FUNCTION_ATTRS
#define HALIDE_FUNCTION_ATTRS
#endif

#ifdef __cplusplus
extern "C" {
#endif

void init_array_1D(buffer_t *buf, int N, uint8_t val);
void print_array_1D(buffer_t *buf, int N);
void print_array_2D(buffer_t buf, int N, int M);
void init_array_val_2D(buffer_t *buf, int N, int M, uint8_t val);
void copy_array_2D(uint8_t* buf, int N, int M, uint8_t* array);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif
