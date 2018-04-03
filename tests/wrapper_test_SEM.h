#ifndef TIRAMISU_SEM_h
#define TIRAMISU_SEM_h


// Define these values for each new test
#define TEST_NAME_STR       "SEM"
#define N 5
//#define UNOPT
#define PRINT_RES

#include <tiramisu/utils.h>

#ifdef __cplusplus
extern "C" {
#endif
  int tiramisu_generated_code(halide_buffer_t*,halide_buffer_t*,halide_buffer_t*);
  int tiramisu_generated_code_argv(void **args);

extern const struct halide_filter_metadata_t halide_pipeline_aot_metadata;
#ifdef __cplusplus
}  // extern "C"
#endif
#endif
