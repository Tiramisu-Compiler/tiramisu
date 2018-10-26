#ifndef HALIDE__build___wrapper_convolution_o_h
#define HALIDE__build___wrapper_convolution_o_h

#include <tiramisu/utils.h>

#define RADIUS 3

#ifdef __cplusplus
extern "C" {
#endif

int convolution_tiramisu(halide_buffer_t *_b_input_buffer, halide_buffer_t *kernel, halide_buffer_t *_b_output_buffer);
int convolution_tiramisu_argv(void **args);
int convolution_ref(halide_buffer_t *_b_input_buffer, halide_buffer_t *kernel, halide_buffer_t *_b_output_buffer);
int convolution_ref_argv(void **args);
// Result is never null and points to constant static data
const struct halide_filter_metadata_t *convolution_tiramisu_metadata();
const struct halide_filter_metadata_t *convolution_ref_metadata();

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
