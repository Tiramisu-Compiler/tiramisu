#ifndef HALIDE__build___wrapper_fusion_o_h
#define HALIDE__build___wrapper_fusion_o_h

#include <coli/utils.h>

#ifdef __cplusplus
extern "C" {
#endif

int fusion_coli(buffer_t *_b_input_buffer, buffer_t *_b_output_f_buffer, buffer_t *_b_output_g_buffer, buffer_t *_b_output_h_buffer, buffer_t *_b_output_k_buffer) HALIDE_FUNCTION_ATTRS;
int fusion_coli_argv(void **args) HALIDE_FUNCTION_ATTRS;
int fusion_ref(buffer_t *_b_input_buffer, buffer_t *_b_output_f_buffer, buffer_t *_b_output_g_buffer, buffer_t *_b_output_h_buffer, buffer_t *_b_output_k_buffer) HALIDE_FUNCTION_ATTRS;
int fusion_ref_argv(void **args) HALIDE_FUNCTION_ATTRS;
// Result is never null and points to constant static data
const struct halide_filter_metadata_t *fusion_coli_metadata() HALIDE_FUNCTION_ATTRS;
const struct halide_filter_metadata_t *fusion_ref_metadata() HALIDE_FUNCTION_ATTRS;

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
