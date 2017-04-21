#ifndef HALIDE__build___generated_lib_tutorial_03_o_h
#define HALIDE__build___generated_lib_tutorial_03_o_h

#include <tiramisu/utils.h>

#ifdef __cplusplus
extern "C" {
#endif

int matmul(halide_buffer_t *b1, halide_buffer_t *b2, halide_buffer_t *b3);
int matmul_argv(void **args);
// Result is never null and points to constant static data
const struct halide_filter_metadata_t *matmul_metadata();

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
