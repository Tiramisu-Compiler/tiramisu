#ifndef TIRAMISU_test_h
#define TIRAMISU_test_h


// Define these values for each new test
#define TEST_NAME_STR       "dynamic_buffer_size"
#define TEST_NUMBER_STR     "87"
// Data size
#define SIZE 10


// --------------------------------------------------------
// No need to modify anything in the following ------------
// --------------------------------------------------------

#include <tiramisu/utils.h>

#ifdef __cplusplus
extern "C" {
#endif
int tiramisu_generated_code(halide_buffer_t *, halide_buffer_t *);
int tiramisu_generated_code_argv(void **args);

extern const struct halide_filter_metadata_t halide_pipeline_aot_metadata;
#ifdef __cplusplus
}  // extern "C"
#endif
#endif
