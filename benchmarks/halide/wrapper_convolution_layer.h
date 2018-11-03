#ifndef HALIDE__build___convolution_layer_o_h
#define HALIDE__build___convolution_layer_o_h

#include <tiramisu/utils.h>

#define RADIUS 3

#ifdef __cplusplus
extern "C" {
#endif

int convolution_layer_tiramisu(halide_buffer_t *parameteres, halide_buffer_t *_b_input_buffer ,halide_buffer_t *filter,halide_buffer_t *bias,halide_buffer_t *_b_output_buffer);
int convolution_layer_tiramisu_argv(void **args);

int convolution_layer_ref(halide_buffer_t *_b_input_buffer ,halide_buffer_t *filter,halide_buffer_t *bias ,halide_buffer_t *_b_output_buffer);
int convolution_layer_ref_argv(void **args);

// Result is never null and points to constant static data
const struct halide_filter_metadata_t *convolution_layer_tiramisu_metadata();
const struct halide_filter_metadata_t *convolution_layer_ref_metadata();

#ifdef __cplusplus
}  // extern "C"
#endif
#endif