#ifndef TIRAMISU_test_h
#define TIRAMISU_test_h


// Define these values for each new test
#define TEST_NAME_STR       "dep_graph"
#define TEST_NUMBER_STR     "23"
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
int tiramisu_generated_code(halide_buffer_t *_p0_buffer, halide_buffer_t *_p1_buffer,
                            halide_buffer_t *_p2_buffer, halide_buffer_t *_p3_buffer, halide_buffer_t *_p4_buffer);
int tiramisu_generated_code_argv(void **args);

extern const struct halide_filter_metadata_t halide_pipeline_aot_metadata;
#ifdef __cplusplus
}  // extern "C"
#endif
#endif
