#ifndef HALIDE__build___wrapper_cvtcolordist_o_h
#define HALIDE__build___wrapper_cvtcolordist_o_h

#include <tiramisu/utils.h>

#ifndef NODES
#define NODES 10
#endif

#define _ROWS 10000
#define _COLS 15000

#ifdef __cplusplus
extern "C" {
#endif

int cvtcolordist_tiramisu(halide_buffer_t *_b_input_buffer, halide_buffer_t *_b_blury_buffer);
int cvtcolordist_ref(halide_buffer_t *_b_input_buffer, halide_buffer_t *_b_blury_buffer);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
