#ifndef HALIDE__build___wrapper_fusion_o_h
#define HALIDE__build___wrapper_fusion_o_h

#include <tiramisu/utils.h>

#ifdef __cplusplus
extern "C" {
#endif

int fusion_tiramisu(halide_buffer_t *_b_input_buffer, halide_buffer_t *_b_output_f_buffer, halide_buffer_t *_b_output_g_buffer, halide_buffer_t *_b_output_h_buffer, halide_buffer_t *_b_output_k_buffer);
int fusion_tiramisu_argv(void **args);
int fusion_ref(halide_buffer_t *_b_input_buffer, halide_buffer_t *_b_output_f_buffer, halide_buffer_t *_b_output_g_buffer, halide_buffer_t *_b_output_h_buffer, halide_buffer_t *_b_output_k_buffer);
int fusion_ref_argv(void **args);
// Result is never null and points to constant static data
const struct halide_filter_metadata_t *fusion_tiramisu_metadata();
const struct halide_filter_metadata_t *fusion_ref_metadata();

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
