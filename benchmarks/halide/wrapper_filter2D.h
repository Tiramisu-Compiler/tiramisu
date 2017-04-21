#ifndef HALIDE__build___wrapper_filter2D_o_h
#define HALIDE__build___wrapper_filter2D_o_h

#include <tiramisu/utils.h>

#define RADIUS 3

#ifdef __cplusplus
extern "C" {
#endif

int filter2D_tiramisu(halide_buffer_t *_b_input_buffer, halide_buffer_t *kernel, halide_buffer_t *_b_output_buffer);
int filter2D_tiramisu_argv(void **args);
int filter2D_ref(halide_buffer_t *_b_input_buffer, halide_buffer_t *kernel, halide_buffer_t *_b_output_buffer);
int filter2D_ref_argv(void **args);
// Result is never null and points to constant static data
const struct halide_filter_metadata_t *filter2D_tiramisu_metadata();
const struct halide_filter_metadata_t *filter2D_ref_metadata();

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
