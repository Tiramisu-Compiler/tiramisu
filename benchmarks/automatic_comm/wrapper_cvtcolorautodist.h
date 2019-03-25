#ifndef HALIDE__build___wrapper_cvtcolorautodist_o_h
#define HALIDE__build___wrapper_cvtcolorautodist_o_h

#include <tiramisu/utils.h>

#ifndef NODES
#define NODES 16
#endif

#define _ROWS 1000
#define _COLS 1500

#ifdef __cplusplus
extern "C" {
#endif

int cvtcolorautodist_tiramisu(halide_buffer_t *_b_input_buffer, halide_buffer_t *_b_blury_buffer);
int cvtcolorautodist_ref(halide_buffer_t *_b_input_buffer, halide_buffer_t *_b_blury_buffer);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
