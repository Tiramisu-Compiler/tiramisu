#ifndef HALIDE__build___wrapper_rgbyuv420_o_h
#define HALIDE__build___wrapper_rgbyuv420_o_h

#include <tiramisu/utils.h>

#ifdef __cplusplus
extern "C" {
#endif

int rgbyuv420_tiramisu(halide_buffer_t *_b_input_buffer, halide_buffer_t *_b_output_y_buffer, halide_buffer_t *_b_output_u_buffer, halide_buffer_t *_b_output_v_buffer);
int rgbyuv420_tiramisu_argv(void **args);
int rgbyuv420_ref(halide_buffer_t *_b_input_buffer, halide_buffer_t *_b_output_y_buffer, halide_buffer_t *_b_output_u_buffer, halide_buffer_t *_b_output_v_buffer);
int rgbyuv420_ref_argv(void **args);
// Result is never null and points to constant static data
const struct halide_filter_metadata_t *rgbyuv420_tiramisu_metadata();
const struct halide_filter_metadata_t *rgbyuv420_ref_metadata();

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
