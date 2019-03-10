#ifndef HALIDE__build___wrapper_convolutiondist_o_h
#define HALIDE__build___wrapper_convolutiondist_o_h

#include <tiramisu/utils.h>

#define RADIUS 3
#define _CHANNELS 3
#define _ROWS 100
#define _COLS 10
#define _NODES 10
#ifdef __cplusplus
extern "C" {
#endif

int convolutiondist_tiramisu(halide_buffer_t *_b_input_buffer, halide_buffer_t *kernel, halide_buffer_t *_b_output_buffer);
int convolutiondist_tiramisu_argv(void **args);
int convolutiondist_ref(halide_buffer_t *_b_input_buffer, halide_buffer_t *kernel, halide_buffer_t *_b_output_buffer);
int convolutiondist_ref_argv(void **args);
// Result is never null and points to constant static data
const struct halide_filter_metadata_t *convolutiondist_tiramisu_metadata();
const struct halide_filter_metadata_t *convolutiondist_ref_metadata();

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
