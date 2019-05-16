#ifndef TIRAMISU_WRAPPER_TEST_1002_H
#define TIRAMISU_WRAPPER_TEST_1002_H

#define TEST_NAME_STR       "Code that elicits bug where splitting loop for MPI Isend (with a shift) causes incorrect buffer address calculation."
#define TEST_NUMBER_STR     "1002"

// Number of MPI ranks.
#define _NUM_RANKS  10
#define _SHIFT_AMOUNT 3

#include <tiramisu/utils.h>

#ifdef __cplusplus
extern "C" {
#endif

int split_isend_bug(halide_buffer_t *_p0_buffer, halide_buffer_t *_p1_buffer, halide_buffer_t *_p2_buffer, halide_buffer_t *_p3_buffer);
int split_isend_bug_argv(void **args);

extern const struct halide_filter_metadata_t halide_pipeline_aot_metadata;

#ifdef __cplusplus
} // extern "C"
#endif


#endif // TIRAMISU_WRAPPER_TEST_1002_H
