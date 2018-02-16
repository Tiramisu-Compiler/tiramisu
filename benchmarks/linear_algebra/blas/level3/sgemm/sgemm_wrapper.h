#ifndef TIRAMISU_test_h
#define TIRAMISU_test_h

// --------------------------------------------------------
// No need to modify anything in the following ------------
// --------------------------------------------------------

#include <tiramisu/utils.h>

#ifdef __cplusplus
extern "C" {
#endif
int tiramisu_generated_code(halide_buffer_t *SIZES, halide_buffer_t *alpha, halide_buffer_t *beta, halide_buffer_t *A, halide_buffer_t *B, halide_buffer_t *C);
int tiramisu_generated_code_argv(void **args);

extern const struct halide_filter_metadata_t halide_pipeline_aot_metadata;
#ifdef __cplusplus
}  // extern "C"
#endif
#endif
