#ifndef TIRAMISU_test_h
#define TIRAMISU_test_h


// Define these values for each new test
#define TEST_NAME_STR       "low_level_separation"
#define TEST_NUMBER_STR     "73"
#define TEST_ID_STR "test_" TEST_NUMBER_STR "_" TEST_NAME_STR
// Data size
#define SIZE 10


// ---------------------------------------------------------------------------------
// TODO: Only need to update the declaration of tiramisu_generated_code ------------
// ---------------------------------------------------------------------------------

#include <tiramisu/utils.h>

#ifdef __cplusplus
extern "C" {
#endif
int tiramisu_generated_code(halide_buffer_t *_p0_buffer);
int tiramisu_generated_code_argv(void **args);

extern const struct halide_filter_metadata_t halide_pipeline_aot_metadata;
#ifdef __cplusplus
}  // extern "C"
#endif
#endif
