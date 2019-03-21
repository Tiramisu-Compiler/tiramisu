#ifndef TIRAMISU_WRAPPER_CONVOLUTION_DIST_DATA_H
#define TIRAMISU_WRAPPER_CONVOLUTION_DIST_DATA_H

#include <tiramisu/utils.h>

#define NCOLS 1000
#define NROWS 1000
#define NNODES 5


#ifdef __cplusplus
extern "C" {
#endif

int convolution_dist(halide_buffer_t *, halide_buffer_t *);
int convolution_dist_argv(void **args);
// Result is never null and points to constant static data
const struct halide_filter_metadata_t *convolution_data_metadata();

#ifdef __cplusplus
}  // extern "C"
#endif


#endif //TIRAMISU_WRAPPER_CONVOLUTION_DIST_DATA_H
