#ifndef TIRAMISU_WRAPPER_TEST_98_H
#define TIRAMISU_WRAPPER_TEST_98_H

// Define these values for each new test
#define TEST_NAME_STR       "Distributed reduction without computation"
#define TEST_NUMBER_STR     "98"
#define REDUC_ITERS 20
// --------------------------------------------------------
// No need to modify anything in the following ------------
// --------------------------------------------------------

#include <tiramisu/utils.h>

#ifdef __cplusplus
extern "C" {
#endif
int dist_comp_only(halide_buffer_t *, halide_buffer_t *);
int dist_comp_only_argv(void **args);

extern const struct halide_filter_metadata_t halide_pipeline_aot_metadata;
#ifdef __cplusplus
}  // extern "C"
#endif

#endif //TIRAMISU_WRAPPER_TEST_98_H
