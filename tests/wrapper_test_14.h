#ifndef HALIDE__build___wrapper_blurxy_test_o_h
#define HALIDE__build___wrapper_blurxy_test_o_h

#include <tiramisu/utils.h>

#ifdef __cplusplus
extern "C" {
#endif

int blurxy_tiramisu_test(halide_buffer_t *_b_input_buffer, halide_buffer_t *_b_blury_buffer);
int blurxy_tiramisu_test_argv(void **args);
// Result is never null and points to constant static data
const struct halide_filter_metadata_t *blurxy_tiramisu_test_metadata();
const struct halide_filter_metadata_t *blurxy_ref_test_metadata();

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
