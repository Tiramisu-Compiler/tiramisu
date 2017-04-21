#ifndef TIRAMISU_test_h
#define TIRAMISU_test_h


// Define these values for each new test
#define TEST_NAME           test_non_affine_accesses
#define TEST_NAME_ARGV      test_non_affine_accesses_argv
#define TEST_NAME_STR       "non_affine_accesses"
#define TEST_NUMBER_STR     "20"
// Data size
#define SIZE0 10
#define SIZE1 10


// --------------------------------------------------------
// No need to modify anything in the following ------------
// --------------------------------------------------------

#include <tiramisu/utils.h>

#ifdef __cplusplus
extern "C" {
#endif
int TEST_NAME(buffer_t *_p0_buffer);
int TEST_NAME_ARGV(void **args);

extern const struct halide_filter_metadata_t halide_pipeline_aot_metadata;
#ifdef __cplusplus
}  // extern "C"
#endif
#endif
