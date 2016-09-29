#ifndef HALIDE__build___generated_lib_tutorial_04_o_h
#define HALIDE__build___generated_lib_tutorial_04_o_h

#include <coli/utils.h>

#ifdef __cplusplus
extern "C" {
#endif

int spmv(buffer_t *b1, buffer_t *b2, buffer_t *b3, buffer_t *b4, buffer_t *b5) HALIDE_FUNCTION_ATTRS;
int spmv_argv(void **args) HALIDE_FUNCTION_ATTRS;
// Result is never null and points to constant static data
const struct halide_filter_metadata_t *spmv_metadata() HALIDE_FUNCTION_ATTRS;

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
