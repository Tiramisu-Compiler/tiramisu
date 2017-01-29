#ifndef HALIDE__build___wrapper_gaussian_o_h
#define HALIDE__build___wrapper_gaussian_o_h

#include <tiramisu/utils.h>

#ifdef __cplusplus
extern "C" {
#endif

int gaussian_coli(buffer_t *_b_input_buffer, buffer_t *kernelX, buffer_t *kernelY, buffer_t *_b_output_buffer) HALIDE_FUNCTION_ATTRS;
int gaussian_coli_argv(void **args) HALIDE_FUNCTION_ATTRS;
int gaussian_ref(buffer_t *_b_input_buffer, buffer_t *kernelX, buffer_t *kernelY, buffer_t *_b_output_buffer) HALIDE_FUNCTION_ATTRS;
int gaussian_ref_argv(void **args) HALIDE_FUNCTION_ATTRS;
// Result is never null and points to constant static data
const struct halide_filter_metadata_t *gaussian_coli_metadata() HALIDE_FUNCTION_ATTRS;
const struct halide_filter_metadata_t *gaussian_ref_metadata() HALIDE_FUNCTION_ATTRS;

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
