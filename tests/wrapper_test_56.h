#ifndef TIRAMISU_test_h
#define TIRAMISU_test_h


// Define these values for each new test
#define TEST_NAME_STR       "clamp_blur"
#define TEST_NUMBER_STR     "56"
// Data size
#define SIZE1 100


// --------------------------------------------------------
// No need to modify anything in the following ------------
// --------------------------------------------------------

#include <tiramisu/utils.h>

#ifdef __cplusplus
extern "C" {
#endif
int blur_100x100_2D_array_with_tiling_parallelism(halide_buffer_t *,halide_buffer_t *);
int blur_100x100_2D_array_with_tiling_parallelism_argv(void **args);

extern const struct halide_filter_metadata_t halide_pipeline_aot_metadata;
#ifdef __cplusplus
}  // extern "C"
#endif
#endif
