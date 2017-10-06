#ifndef TIRAMISU_test_h
#define TIRAMISU_test_h


// Define these values for each new test
#define TEST_NAME_STR       "variable_size_matrix_mult"
#define TEST_NUMBER_STR     "85"
#define TEST_ID_STR "test_" TEST_NUMBER_STR "_" TEST_NAME_STR
// Data size

#define N1 7
#define D1 7
#define M1 3

#define N2 1023
#define D2 234
#define M2 4523

// ---------------------------------------------------------------------------------
// TODO: Only need to update the declaration of tiramisu_generated_code ------------
// ---------------------------------------------------------------------------------

#include <tiramisu/utils.h>

#ifdef __cplusplus
extern "C" {
#endif
int tiramisu_generated_code(halide_buffer_t *, halide_buffer_t *,
                            halide_buffer_t *, halide_buffer_t *);
int tiramisu_generated_code_argv(void **args);

extern const struct halide_filter_metadata_t halide_pipeline_aot_metadata;
#ifdef __cplusplus
}  // extern "C"
#endif
#endif
