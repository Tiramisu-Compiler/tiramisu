#ifndef HALIDE__build___wrapper_cvtcolor_o_h
#define HALIDE__build___wrapper_cvtcolor_o_h

#include <tiramisu/utils.h>

#ifdef __cplusplus
extern "C" {
#endif

int cvtcolor_tiramisu(halide_buffer_t *_b_input_buffer, halide_buffer_t *_b_blury_buffer);
int cvtcolor_tiramisu_argv(void **args);
int cvtcolor_ref(halide_buffer_t *_b_input_buffer, halide_buffer_t *_b_blury_buffer);
int cvtcolor_ref_argv(void **args);
// Result is never null and points to constant static data
const struct halide_filter_metadata_t *cvtcolor_tiramisu_metadata();
const struct halide_filter_metadata_t *cvtcolor_ref_metadata();

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
