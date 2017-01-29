#ifndef HALIDE__build___generated_lib_tutorial_03_o_h
#define HALIDE__build___generated_lib_tutorial_03_o_h

#include <tiramisu/utils.h>

#ifdef __cplusplus
extern "C" {
#endif

int matmul(buffer_t *b1, buffer_t *b2, buffer_t *b3) HALIDE_FUNCTION_ATTRS;
int matmul_argv(void **args) HALIDE_FUNCTION_ATTRS;
// Result is never null and points to constant static data
const struct halide_filter_metadata_t *matmul_metadata() HALIDE_FUNCTION_ATTRS;

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
