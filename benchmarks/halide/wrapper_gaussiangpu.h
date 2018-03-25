#ifndef HALIDE__build___wrapper_gaussian_o_h
#define HALIDE__build___wrapper_gaussian_o_h

#include <tiramisu/utils.h>

#ifdef __cplusplus
extern "C" {
#endif

int gaussian_tiramisu(halide_buffer_t *, halide_buffer_t *_b_input_buffer, halide_buffer_t *kernelX, halide_buffer_t *kernelY, halide_buffer_t *_b_output_buffer);
int gaussian_tiramisu_argv(void **args);
int gaussian_ref(halide_buffer_t *_b_input_buffer, halide_buffer_t *kernelX, halide_buffer_t *kernelY, halide_buffer_t *_b_output_buffer);
int gaussian_ref_argv(void **args);
// Result is never null and points to constant static data
const struct halide_filter_metadata_t *gaussian_tiramisu_metadata();
const struct halide_filter_metadata_t *gaussian_ref_metadata();

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
