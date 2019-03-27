//
// Created by Jessica Ray on 11/2/17.
//

#ifndef TIRAMISU_WRAPPER_CVTCOLOR_DIST_DATA_H
#define TIRAMISU_WRAPPER_CVTCOLOR_DIST_DATA_H

#include <tiramisu/utils.h>

#define NROWS 100000
#define NCOLS 50000

#ifdef __cplusplus
extern "C" {
#endif

int cvtcolor_dist(halide_buffer_t *, halide_buffer_t *);
int cvtcolor_dist_argv(void **args);
// Result is never null and points to constant static data
const struct halide_filter_metadata_t *cvtcolor_data_metadata();

#ifdef __cplusplus
}  // extern "C"
#endif


#endif //TIRAMISU_WRAPPER_CVTCOLOR_DIST_DATA_H
