#ifndef HALIDE__build___wrapper_heat2dgpu_o_h
#define HALIDE__build___wrapper_heat2dgpu_o_h

#include <tiramisu/utils.h>

#ifdef __cplusplus
extern "C" {
#endif

int heat2dgpu_tiramisu(halide_buffer_t *_b_input_buffer, halide_buffer_t *_b_output_buffer);
int heat2dgpu_tiramisu_argv(void **args);
int heat2dgpu_ref(halide_buffer_t *_b_input_buffer, halide_buffer_t *_b_output_buffer);
int heat2dgpu_ref_argv(void **args);
// Result is never null and points to constant static data
const struct halide_filter_metadata_t *heat2dgpu_tiramisu_metadata();
const struct halide_filter_metadata_t *heat2dgpu_ref_metadata();

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
