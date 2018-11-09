#ifndef HALIDE__build___wrapper_optical_flow_o_h
#define HALIDE__build___wrapper_optical_flow_o_h

#include <tiramisu/utils.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SYNTHETIC_INPUT 1

#ifdef SYNTHETIC_INPUT
    // Window size
    #define w 2
    #define _NC 2
#else
    // Window size
    #define w 128
    #define _NC 8
#endif

// Corner number

// Number of pyramid levels
#define npyramids 3

int py_optical_flow_tiramisu(halide_buffer_t *, halide_buffer_t *, halide_buffer_t *,
			  halide_buffer_t *, halide_buffer_t *, halide_buffer_t *,
			  halide_buffer_t *, halide_buffer_t *, halide_buffer_t *, halide_buffer_t *, halide_buffer_t *, halide_buffer_t *, halide_buffer_t *, halide_buffer_t *i, halide_buffer_t *, halide_buffer_t *, halide_buffer_t *, halide_buffer_t *);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
