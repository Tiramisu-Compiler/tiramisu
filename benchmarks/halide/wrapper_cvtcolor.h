#ifndef HALIDE__build___wrapper_cvt_color_o_h
#define HALIDE__build___wrapper_cvt_color_o_h

#include <coli/utils.h>

#ifdef __cplusplus
extern "C" {
#endif

int cvtcolor_coli(buffer_t *_b_input_buffer, buffer_t *_b_blury_buffer) HALIDE_FUNCTION_ATTRS;
int cvtcolor_coli_argv(void **args) HALIDE_FUNCTION_ATTRS;
int cvtcolor_ref(buffer_t *_b_input_buffer, buffer_t *_b_blury_buffer) HALIDE_FUNCTION_ATTRS;
int cvtcolor_ref_argv(void **args) HALIDE_FUNCTION_ATTRS;
// Result is never null and points to constant static data
const struct halide_filter_metadata_t *cvtcolor_coli_metadata() HALIDE_FUNCTION_ATTRS;
const struct halide_filter_metadata_t *cvtcolor_ref_metadata() HALIDE_FUNCTION_ATTRS;

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
