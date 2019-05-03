#ifndef TIRAMISU_WRAPPER_TEST_1001_H
#define TIRAMISU_WRAPPER_TEST_1001_H

#define TEST_NAME_STR       "Code that elicits bug where more variables tagged distributed than requested."
#define TEST_NUMBER_STR     "1001"

// Number of MPI ranks.
#define _NUM_RANKS  10

#include <tiramisu/utils.h>

#ifdef __cplusplus
extern "C" {
#endif

int double_distribute_bug(halide_buffer_t *_p0_buffer, halide_buffer_t *_p1_buffer, halide_buffer_t *_p2_buffer, halide_buffer_t *_p3_buffer);
int double_distribute_bug_argv(void **args);

extern const struct halide_filter_metadata_t halide_pipeline_aot_metadata;

#ifdef __cplusplus
} // extern "C"
#endif


#endif // TIRAMISU_WRAPPER_TEST_1001_H
