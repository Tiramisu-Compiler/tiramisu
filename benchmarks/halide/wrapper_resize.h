#ifndef HALIDE__build___wrapper_resize_o_h
#define HALIDE__build___wrapper_resize_o_h

#include <tiramisu/utils.h>

#ifdef __cplusplus
extern "C" {
#endif

int resize_tiramisu(halide_buffer_t *_b_input_buffer, halide_buffer_t *_b_blury_buffer);
int resize_tiramisu_argv(void **args);
int resize_ref(halide_buffer_t *_b_input_buffer, halide_buffer_t *_b_blury_buffer);
int resize_ref_argv(void **args);
// Result is never null and points to constant static data
const struct halide_filter_metadata_t *resize_tiramisu_metadata();
const struct halide_filter_metadata_t *resize_ref_metadata();

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
