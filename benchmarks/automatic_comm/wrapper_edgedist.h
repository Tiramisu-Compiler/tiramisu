#ifndef HALIDE__build___wrapper_edgedist_o_h
#define HALIDE__build___wrapper_edgedist_o_h

#include <tiramisu/utils.h>
#ifdef __cplusplus
extern "C" {
#endif

#define _ROWS 1500
#define _COLS 1000

#define NODES 10

int edgedist_tiramisu(halide_buffer_t *, halide_buffer_t *);
int edgedist_ref(halide_buffer_t *, halide_buffer_t *);

#ifdef __cplusplus
}
#endif
#endif
