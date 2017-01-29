#ifndef HALIDE__build___wrapper_rgbyuv420_o_h
#define HALIDE__build___wrapper_rgbyuv420_o_h

#include <tiramisu/utils.h>

#ifdef __cplusplus
extern "C" {
#endif

int rgbyuv420_coli(buffer_t *_b_input_buffer, buffer_t *_b_output_y_buffer, buffer_t *_b_output_u_buffer, buffer_t *_b_output_v_buffer) HALIDE_FUNCTION_ATTRS;
int rgbyuv420_coli_argv(void **args) HALIDE_FUNCTION_ATTRS;
int rgbyuv420_ref(buffer_t *_b_input_buffer, buffer_t *_b_output_y_buffer, buffer_t *_b_output_u_buffer, buffer_t *_b_output_v_buffer) HALIDE_FUNCTION_ATTRS;
int rgbyuv420_ref_argv(void **args) HALIDE_FUNCTION_ATTRS;
// Result is never null and points to constant static data
const struct halide_filter_metadata_t *rgbyuv420_coli_metadata() HALIDE_FUNCTION_ATTRS;
const struct halide_filter_metadata_t *rgbyuv420_ref_metadata() HALIDE_FUNCTION_ATTRS;

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
