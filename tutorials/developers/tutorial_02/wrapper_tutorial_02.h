#ifndef HALIDE__build___generated_lib_tutorial_02_o_h
#define HALIDE__build___generated_lib_tutorial_02_o_h

#include <tiramisu/utils.h>

#ifdef __cplusplus
extern "C" {
#endif

int blurxy(halide_buffer_t *_b_input_buffer, halide_buffer_t *_b_blury_buffer);
int blurxy_argv(void **args);
// Result is never null and points to constant static data
const struct halide_filter_metadata_t *blurxy_metadata();

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
