#ifndef HALIDE__build___wrapper_convolutiongpu_o_h
#define HALIDE__build___wrapper_convolutiongpu_o_h

#include <tiramisu/utils.h>

#define RADIUS 3

#ifdef __cplusplus
extern "C" {
#endif

int convolutiongpu_tiramisu(halide_buffer_t *_b_input_buffer, halide_buffer_t *kernel, halide_buffer_t *_b_output_buffer);
int convolutiongpu_tiramisu_argv(void **args);
int convolutiongpu_ref(halide_buffer_t *_b_input_buffer, halide_buffer_t *kernel, halide_buffer_t *_b_output_buffer);
int convolutiongpu_ref_argv(void **args);
// Result is never null and points to constant static data
const struct halide_filter_metadata_t *convolutiongpu_tiramisu_metadata();
const struct halide_filter_metadata_t *convolutiongpu_ref_metadata();

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
