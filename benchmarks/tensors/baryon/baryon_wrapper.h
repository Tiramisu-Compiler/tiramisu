#ifndef TIRAMISU_test_h
#define TIRAMISU_test_h


// Define these values for each new test
#define TEST_NAME_STR       "baryon"

// Data size
#if TIRAMISU_XLARGE
#define SIZE (1024*1024*128)
#elif TIRAMISU_LARGE
#define SIZE (1024*1024)
#elif TIRAMISU_MEDIUM
#define SIZE (1024)
#elif TIRAMISU_SMALL
#define SIZE (128)
#endif

// --------------------------------------------------------
// No need to modify anything in the following ------------
// --------------------------------------------------------

#include <tiramisu/utils.h>

#ifdef __cplusplus
extern "C" {
#endif
int tiramisu_generated_code(halide_buffer_t *_p0_buffer, halide_buffer_t *_p1_buffer, halide_buffer_t *_p2_buffer);
int tiramisu_generated_code_argv(void **args);

extern const struct halide_filter_metadata_t halide_pipeline_aot_metadata;
#ifdef __cplusplus
}  // extern "C"
#endif
#endif
