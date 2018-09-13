#ifndef HALIDE__build___wrapper_edgegpu_o_h
#define HALIDE__build___wrapper_edgegpu_o_h

#include <tiramisu/utils.h>

#ifdef __cplusplus
extern "C" {
#endif

int edge_tiramisu(halide_buffer_t *, halide_buffer_t *);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
