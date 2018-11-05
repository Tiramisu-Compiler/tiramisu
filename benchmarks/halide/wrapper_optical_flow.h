#ifndef HALIDE__build___wrapper_optical_flow_o_h
#define HALIDE__build___wrapper_optical_flow_o_h

#include <tiramisu/utils.h>

#ifdef __cplusplus
extern "C" {
#endif

// Corner number
#define _NC 8

// Window size
#define w 128

int optical_flow_tiramisu(halide_buffer_t *, halide_buffer_t *, halide_buffer_t *,
			  halide_buffer_t *, halide_buffer_t *, halide_buffer_t *,
			  halide_buffer_t *, halide_buffer_t *, halide_buffer_t *, halide_buffer_t *, halide_buffer_t *, halide_buffer_t *, halide_buffer_t *, halide_buffer_t *i, halide_buffer_t *, halide_buffer_t *);
//int warp_affine_ref(halide_buffer_t *_b_input_buffer, halide_buffer_t *_b_blury_buffer);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
