#ifndef HALIDE__build___wrapper_warp_affinegpu_o_h
#define HALIDE__build___wrapper_warp_affinegpu_o_h

#include <tiramisu/utils.h>

#ifdef __cplusplus
extern "C" {
#endif

int warp_affinegpu_tiramisu(halide_buffer_t *_b_input_buffer, halide_buffer_t *_b_blury_buffer);
int warp_affinegpu_tiramisu_argv(void **args);
int warp_affinegpu_ref(halide_buffer_t *_b_input_buffer, halide_buffer_t *_b_blury_buffer);
int warp_affinegpu_ref_argv(void **args);
// Result is never null and points to constant static data
const struct halide_filter_metadata_t *warp_affinegpu_tiramisu_metadata();
const struct halide_filter_metadata_t *warp_affinegpu_ref_metadata();

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
