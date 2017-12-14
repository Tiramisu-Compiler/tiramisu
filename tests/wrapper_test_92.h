#ifndef TIRAMISU_test_h
#define TIRAMISU_test_h


// Define these values for each new test
#define TEST_NAME_STR       ""
#define TEST_NUMBER_STR     "92"
#define TEST_ID_STR "test_" TEST_NUMBER_STR "_" TEST_NAME_STR
// Data size
// TODO: define any data sizes here


// ---------------------------------------------------------------------------------
// TODO: Only need to update the declaration of tiramisu_generated_code ------------
// ---------------------------------------------------------------------------------

#include <tiramisu/utils.h>

#ifdef __cplusplus
extern "C" {
#endif
int tiramisu_generated_code(/* TODO: add "halide_buffer_t *" for every input/output buffer*/);
int tiramisu_generated_code_argv(void **args);

extern const struct halide_filter_metadata_t halide_pipeline_aot_metadata;
#ifdef __cplusplus
}  // extern "C"
#endif
#endif
