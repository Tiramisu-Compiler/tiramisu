#ifndef TIRAMISU_WRAPPER_TEST_1000_H
#define TIRAMISU_WRAPPER_TEST_1000_H

#define TEST_NAME_STR       "Distributed dense matrix-matrix multiplication"
#define TEST_NUMBER_STR     "1000"

// Number of MPI ranks and matrix dimension sizes.
#define _NUM_RANKS  72
#define _I_DIM_SIZE 50
#define _J_DIM_SIZE 50
#define _K_DIM_SIZE 50

#include <tiramisu/utils.h>

#ifdef __cplusplus
extern "C" {
#endif

  int spmm(halide_buffer_t *_p0_buffer, halide_buffer_t *_p1_buffer, halide_buffer_t *_p2_buffer, halide_buffer_t *_p3_buffer, halide_buffer_t *_p4_buffer);
  int spmm_argv(void **args);
  
  extern const struct halide_filter_metadata_t halide_pipeline_aot_metadata;

#ifdef __cplusplus
} // extern "C"
#endif


#endif // TIRAMISU_WRAPPER_TEST_1000_H
