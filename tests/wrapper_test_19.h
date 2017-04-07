#ifndef TIRAMISU_test_h
#define TIRAMISU_test_h


// Define these values for each new test
#define TEST_NAME           test_gpu_tile_3D
#define TEST_NAME_ARGV      test_gpu_tile_3D_argv
#define TEST_NAME_STR       "gpu_tile_3D"
#define TEST_NUMBER_STR     "19"
// Data size
#define SIZE0 4
#define SIZE1 4


// --------------------------------------------------------
// No need to modify anything in the following ------------
// --------------------------------------------------------

#include <tiramisu/utils.h>

#ifdef __cplusplus
extern "C" {
#endif
int TEST_NAME(buffer_t *_p0_buffer) HALIDE_FUNCTION_ATTRS;
int TEST_NAME_ARGV(void **args) HALIDE_FUNCTION_ATTRS;

extern const struct halide_filter_metadata_t halide_pipeline_aot_metadata;
#ifdef __cplusplus
}  // extern "C"
#endif
#endif
