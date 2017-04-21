#ifndef HALIDE__build___wrapper_divergence2d_o_h
#define HALIDE__build___wrapper_divergence2d_o_h

#include <tiramisu/utils.h>

#ifdef __cplusplus
extern "C" {
#endif

int divergence2d_tiramisu(halide_buffer_t *_b_input_buffer, halide_buffer_t *_b_blury_buffer);
int divergence2d_tiramisu_argv(void **args);
int divergence2d_ref(halide_buffer_t *_b_input_buffer, halide_buffer_t *_b_blury_buffer);
int divergence2d_ref_argv(void **args);
// Result is never null and points to constant static data
const struct halide_filter_metadata_t *divergence2d_tiramisu_metadata();
const struct halide_filter_metadata_t *divergence2d_ref_metadata();

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
