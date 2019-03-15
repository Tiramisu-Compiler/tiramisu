#ifndef HALIDE__build___wrapper_convolutionautodist_o_h
#define HALIDE__build___wrapper_convolutionautodist_o_h

#include <tiramisu/utils.h>

#define RADIUS 3
#define _CHANNELS 3
#define _ROWS 1500
#define _COLS 1000
#define _NODES 10

#ifdef __cplusplus
extern "C" {
#endif

int convolutionautodist_tiramisu(halide_buffer_t *_b_input_buffer, halide_buffer_t *kernel, halide_buffer_t *_b_output_buffer);
int convolutionautodist_ref(halide_buffer_t *_b_input_buffer, halide_buffer_t *kernel, halide_buffer_t *_b_output_buffer);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
