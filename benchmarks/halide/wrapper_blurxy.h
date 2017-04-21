#ifndef HALIDE__build___wrapper_blurxy_o_h
#define HALIDE__build___wrapper_blurxy_o_h

#include <tiramisu/utils.h>

#ifdef __cplusplus
extern "C" {
#endif

int blurxy_tiramisu(halide_buffer_t *_b_input_buffer, halide_buffer_t *_b_blury_buffer);
int blurxy_tiramisu_argv(void **args);
int blurxy_ref(halide_buffer_t *_b_input_buffer, halide_buffer_t *_b_blury_buffer);
int blurxy_ref_argv(void **args);
// Result is never null and points to constant static data
const struct halide_filter_metadata_t *blurxy_tiramisu_metadata();
const struct halide_filter_metadata_t *blurxy_ref_metadata();

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
