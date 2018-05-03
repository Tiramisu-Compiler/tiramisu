#ifndef TIRAMISU_WRAPPER_TEST_100_H
#define TIRAMISU_WRAPPER_TEST_100_H

// Define these values for each new test
#define TEST_NAME_STR       "Distributed communication only (non-blocking)"
#define TEST_NUMBER_STR     "100"

// --------------------------------------------------------
// No need to modify anything in the following ------------
// --------------------------------------------------------

#include <tiramisu/utils.h>

#ifdef __cplusplus
extern "C" {
#endif
int dist_comm_only_nonblock(halide_buffer_t *);
int dist_comm_only_nonblock_argv(void **args);

extern const struct halide_filter_metadata_t halide_pipeline_aot_metadata;
#ifdef __cplusplus
}  // extern "C"


#endif //TIRAMISU_WRAPPER_TEST_100_H
