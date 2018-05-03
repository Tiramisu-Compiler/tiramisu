#ifndef HALIDE__build___wrapper_fusiongpu_o_h
#define HALIDE__build___wrapper_fusiongpu_o_h

#include <tiramisu/utils.h>

#ifdef __cplusplus
extern "C" {
#endif

int fusiongpu_tiramisu(halide_buffer_t *, halide_buffer_t *_b_input_buffer, halide_buffer_t *_b_output_f_buffer, halide_buffer_t *_b_output_g_buffer, halide_buffer_t *_b_output_h_buffer, halide_buffer_t *_b_output_k_buffer);
int fusiongpu_tiramisu_argv(void **args);
int fusiongpu_ref(halide_buffer_t *_b_input_buffer, halide_buffer_t *_b_output_f_buffer, halide_buffer_t *_b_output_g_buffer, halide_buffer_t *_b_output_h_buffer, halide_buffer_t *_b_output_k_buffer);
int fusiongpu_ref_argv(void **args);
// Result is never null and points to constant static data
const struct halide_filter_metadata_t *fusiongpu_tiramisu_metadata();
const struct halide_filter_metadata_t *fusiongpu_ref_metadata();

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
