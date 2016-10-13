#ifndef HALIDE__build___generated_lib_tutorial_05_o_h
#define HALIDE__build___generated_lib_tutorial_05_o_h

#include <coli/utils.h>

#ifdef __cplusplus
extern "C" {
#endif

int sequence(buffer_t *b1, buffer_t *b2, buffer_t *b3, buffer_t *b4) HALIDE_FUNCTION_ATTRS;
int sequence_argv(void **args) HALIDE_FUNCTION_ATTRS;
// Result is never null and points to constant static data
const struct halide_filter_metadata_t *spmv_metadata() HALIDE_FUNCTION_ATTRS;

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
