#ifndef HALIDE__build___wrapper_heat3d_o_h
#define HALIDE__build___wrapper_heat3d_o_h

//dimensions
#define _X 100
#define _Y 110
#define _Z 120
//time
#define _TIME 200
//the constants of the algorithm
#define _ALPHA 0.125f
#define _BETA 2.0f
//constants
#define _BASE 10

#include <tiramisu/utils.h>

#ifdef __cplusplus
extern "C" {
#endif
int heat3d_tiramisu(halide_buffer_t *_input_buffer, halide_buffer_t *_out_buffer);
int heat3d_ref(halide_buffer_t *_input_buffer, halide_buffer_t *_out_buffer);
int heat3d_ref_argv(void **args);
#ifdef __cplusplus
}  // extern "C"
#endif
#endif
