#ifndef HALIDE__build___generated_lib_tutorial_02_o_h
#define HALIDE__build___generated_lib_tutorial_02_o_h

#include <coli/utils.h>

#ifdef __cplusplus
extern "C" {
#endif

int blurxy(buffer_t *_b_input_buffer, buffer_t *_b_blury_buffer) HALIDE_FUNCTION_ATTRS;
int blurxy_argv(void **args) HALIDE_FUNCTION_ATTRS;
// Result is never null and points to constant static data
const struct halide_filter_metadata_t *blurxy_metadata() HALIDE_FUNCTION_ATTRS;

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
