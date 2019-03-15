#ifndef HALIDE__build___wrapper_edgeautodist_o_h
#define HALIDE__build___wrapper_edgeautodist_o_h

#include <tiramisu/utils.h>
#ifdef __cplusplus
extern "C" {
#endif

#define _ROWS 1500
#define _COLS 1000

#define NODES 10

int edgeautodist_tiramisu(halide_buffer_t *, halide_buffer_t *);
int edgeautodist_ref(halide_buffer_t *, halide_buffer_t *);

#ifdef __cplusplus
}
#endif
#endif
