#ifndef TIRAMISU_WRAPPER_TEST_99_H
#define TIRAMISU_WRAPPER_TEST_99_H

// Define these values for each new test
#define TEST_NAME_STR       "Distributed communication only"
#define TEST_NUMBER_STR     "99"

// --------------------------------------------------------
// No need to modify anything in the following ------------
// --------------------------------------------------------

#include <tiramisu/utils.h>

#ifdef __cplusplus
extern "C" {
#endif
int dist_comm_only(halide_buffer_t *);
int dist_comm_only_argv(void **args);

extern const struct halide_filter_metadata_t halide_pipeline_aot_metadata;
#ifdef __cplusplus
}  // extern "C"
#endif

#endif //TIRAMISU_WRAPPER_TEST_99_H
