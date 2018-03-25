#ifndef HALIDE__build___wrapper_cvtcolorgpu_o_h
#define HALIDE__build___wrapper_cvtcolorgpu_o_h

#include <tiramisu/utils.h>

#ifdef __cplusplus
extern "C" {
#endif

int cvtcolorgpu_tiramisu(halide_buffer_t *SIZES, halide_buffer_t *_b_input_buffer, halide_buffer_t *_b_blury_buffer);
int cvtcolorgpu_tiramisu_argv(void **args);
int cvtcolorgpu_ref(halide_buffer_t *_b_input_buffer, halide_buffer_t *_b_blury_buffer);
int cvtcolorgpu_ref_argv(void **args);
// Result is never null and points to constant static data
const struct halide_filter_metadata_t *cvtcolorgpu_tiramisu_metadata();
const struct halide_filter_metadata_t *cvtcolorgpu_ref_metadata();

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
