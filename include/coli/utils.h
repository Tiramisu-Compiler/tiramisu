#ifndef _COLI_UTILS
#define _COLI_UTILS

#include <string>
#include <chrono>
#include <vector>

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

double median(std::vector<std::chrono::duration<double,std::milli>> scores);
void print_time(std::string file_name, std::string kernel_name,
                std::vector<std::string> header_text,
                std::vector<double>      time_vector);
void init_1D_buffer(buffer_t *buf, int N, uint8_t val);
void print_1D_buffer(buffer_t *buf, int N);
void init_1D_buffer_val(buffer_t *buf, int N, uint8_t val);
buffer_t allocate_1D_buffer(int NN);
void print_2D_array(buffer_t buf, int N, int M);
void init_2D_buffer_val(buffer_t *buf, int N, int M, uint8_t val);
void copy_2D_buffer(uint8_t* buf, int N, int M, uint8_t* array);
buffer_t allocate_2D_buffer(int NN, int MM);
void compare_2_2D_arrays(std::string str, uint8_t *array1, uint8_t *array2, int N, int M);
void compare_2_1D_arrays(std::string str, uint8_t *array1, uint8_t *array2, int N);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif
