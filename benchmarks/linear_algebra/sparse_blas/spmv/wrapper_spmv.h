#ifndef HALIDE__build___generated_lib_tutorial_04_o_h
#define HALIDE__build___generated_lib_tutorial_04_o_h

#include <tiramisu/utils.h>

#ifdef __cplusplus
extern "C" {
#endif

int spmv(halide_buffer_t *b1, halide_buffer_t *b2, halide_buffer_t *b3, halide_buffer_t *b4,
         halide_buffer_t *b5);
int spmv_argv(void **args);
// Result is never null and points to constant static data
const struct halide_filter_metadata_t *spmv_metadata();

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
