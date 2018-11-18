#ifndef HALIDE__build___wrapper_heat3ddist_o_h
#define HALIDE__build___wrapper_heat3ddist_o_h

//dimensions
#define _X 30
#define _Y 30
#define _Z 400
#ifndef NODES
#define NODES 4
#endif
//time
#define _TIME 200
//the constants of the algorithm
#define _ALPHA 0.125f
#define _BETA 2.0f

#include <tiramisu/utils.h>

#ifdef __cplusplus
extern "C" {
#endif
int heat3ddist(halide_buffer_t *_input_buffer, halide_buffer_t *_out_buffer);
int heat3ddist_ref(halide_buffer_t *_input_buffer, halide_buffer_t *_out_buffer);
int heat3ddist_ref_argv(void **args);
#ifdef __cplusplus
}  // extern "C"
#endif
#endif
