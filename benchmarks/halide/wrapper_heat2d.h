#ifndef HALIDE__build___wrapper_heat2d_o_h
#define HALIDE__build___wrapper_heat2d_o_h

#include <coli/utils.h>

#ifdef __cplusplus
extern "C" {
#endif

int heat2d_coli(buffer_t *_b_input_buffer, buffer_t *_b_output_buffer) HALIDE_FUNCTION_ATTRS;
int heat2d_coli_argv(void **args) HALIDE_FUNCTION_ATTRS;
int heat2d_ref(buffer_t *_b_input_buffer, buffer_t *_b_output_buffer) HALIDE_FUNCTION_ATTRS;
int heat2d_ref_argv(void **args) HALIDE_FUNCTION_ATTRS;
// Result is never null and points to constant static data
const struct halide_filter_metadata_t *heat2d_coli_metadata() HALIDE_FUNCTION_ATTRS;
const struct halide_filter_metadata_t *heat2d_ref_metadata() HALIDE_FUNCTION_ATTRS;

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
