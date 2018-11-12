#ifndef HALIDE__build___wrapper_optical_flow_o_h
#define HALIDE__build___wrapper_optical_flow_o_h

#include <tiramisu/utils.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SYNTHETIC_INPUT 1
#define SYNTHETIC_INPUT_SIZE 20

#ifdef SYNTHETIC_INPUT
    // Window size
    #define w 4
#else
    // Window size
    #define w 128
#endif

// Corner number

// Number of pyramid levels
#define npyramids 2

// Number of refinement iterations
#define niterations 1

int py_optical_flow_tiramisu(halide_buffer_t *, halide_buffer_t *, halide_buffer_t *,
			  halide_buffer_t *, halide_buffer_t *, halide_buffer_t *,
			  halide_buffer_t *, halide_buffer_t *, halide_buffer_t *, halide_buffer_t *, halide_buffer_t *, halide_buffer_t *, halide_buffer_t *, halide_buffer_t *i, halide_buffer_t *, halide_buffer_t *, halide_buffer_t *);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
