#ifndef HALIDE__build___wrapper_vgg_o_h
#define HALIDE__build___wrapper_vgg_o_h

#include <tiramisu/utils.h>

#define RADIUS 3

#ifdef __cplusplus
extern "C" {
#endif

int vgg_tiramisu(halide_buffer_t *parameteres, halide_buffer_t *_b_input_buffer ,halide_buffer_t *filter,halide_buffer_t *bias,halide_buffer_t *conv,halide_buffer_t *filter2, halide_buffer_t *bias2 ,halide_buffer_t *conv2,halide_buffer_t *_b_output_buffer,halide_buffer_t *_negative_slope);
int vgg_tiramisu_argv(void **args);

int vgg_ref(halide_buffer_t *_b_input_buffer ,halide_buffer_t *filter,halide_buffer_t *bias, halide_buffer_t *filter2, halide_buffer_t *bias2 ,halide_buffer_t *_b_output_buffer);
int vgg_ref_argv(void **args);

// Result is never null and points to constant static data
const struct halide_filter_metadata_t *vgg_tiramisu_metadata();
const struct halide_filter_metadata_t *vgg_ref_metadata();

#ifdef __cplusplus
}  // extern "C"
#endif
#endif