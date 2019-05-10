#ifndef TIRAMISU_WRAPPER_TEST_1003_H
#define TIRAMISU_WRAPPER_TEST_1003_H

#define TEST_NAME_STR       "Object API matrix-matrix multiplication"
#define TEST_NUMBER_STR     "1003"

// Matrix dimension sizes.
#define _I_DIM_SIZE 6
#define _J_DIM_SIZE 10
#define _K_DIM_SIZE 14

#include <tiramisu/utils.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int32_t dim0;
  int32_t dim1;
  uint64_t data[];
} obj_mat;

int mm_obj(halide_buffer_t *_p0_buffer, halide_buffer_t *_p1_buffer, halide_buffer_t *_p2_buffer, halide_buffer_t *_p3_buffer);
int mm_obj_argv(void **args);

extern const struct halide_filter_metadata_t halide_pipeline_aot_metadata;

#ifdef __cplusplus
} // extern "C"
#endif


#endif // TIRAMISU_WRAPPER_TEST_1003_H
