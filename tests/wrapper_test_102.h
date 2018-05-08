#ifndef TIRAMISU_WRAPPER_TEST_102_H
#define TIRAMISU_WRAPPER_TEST_102_H

// Define these values for each new test
#define TEST_NAME_STR       "Distributed reduction collapse"
#define TEST_NUMBER_STR     "102"

// --------------------------------------------------------
// No need to modify anything in the following ------------
// --------------------------------------------------------

#include <tiramisu/utils.h>

#ifdef __cplusplus
extern "C" {
#endif
int dist_reduction_collapse(halide_buffer_t *, halide_buffer_t *);
int dist_reduction_argv_collapse(void **args);

extern const struct halide_filter_metadata_t halide_pipeline_aot_metadata;
#ifdef __cplusplus
}  // extern "C"
#endif

#endif //TIRAMISU_WRAPPER_TEST_102_H
