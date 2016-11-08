#ifndef HALIDE__build___wrapper_filter2D_o_h
#define HALIDE__build___wrapper_filter2D_o_h

#include <coli/utils.h>

#define RADIUS 3

#ifdef __cplusplus
extern "C" {
#endif

int filter2D_coli(buffer_t *_b_input_buffer, buffer_t *kernel, buffer_t *_b_output_buffer) HALIDE_FUNCTION_ATTRS;
int filter2D_coli_argv(void **args) HALIDE_FUNCTION_ATTRS;
int filter2D_ref(buffer_t *_b_input_buffer, buffer_t *kernel, buffer_t *_b_output_buffer) HALIDE_FUNCTION_ATTRS;
int filter2D_ref_argv(void **args) HALIDE_FUNCTION_ATTRS;
// Result is never null and points to constant static data
const struct halide_filter_metadata_t *filter2D_coli_metadata() HALIDE_FUNCTION_ATTRS;
const struct halide_filter_metadata_t *filter2D_ref_metadata() HALIDE_FUNCTION_ATTRS;

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
