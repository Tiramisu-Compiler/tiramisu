//
// Created by Jessica Ray on 11/2/17.
//

#ifndef TIRAMISU_WRAPPER_BLURXY_DIST_H
#define TIRAMISU_WRAPPER_BLURXY_DIST_H

#include <tiramisu/utils.h>

#define NCOLS 1000
#define NROWS 1000
#define NNODES 5

#ifdef __cplusplus
extern "C" {
#endif

  int blurxy_dist(halide_buffer_t *, halide_buffer_t *,  halide_buffer_t *);
int blurxy_dist_argv(void **args);
// Result is never null and points to constant static data
const struct halide_filter_metadata_t *blurxy_dist_metadata();

#ifdef __cplusplus
}  // extern "C"
#endif


#endif //TIRAMISU_WRAPPER_BLURXY_DIST_H
