#ifndef HALIDE__build___wrapper_recfilter_o_h
#define HALIDE__build___wrapper_recfilter_o_h

#include <tiramisu/utils.h>

#ifdef __cplusplus
extern "C" {
#endif

int recfilter_coli(buffer_t *_b_input_buffer, buffer_t *_b_blury_buffer) HALIDE_FUNCTION_ATTRS;
int recfilter_coli_argv(void **args) HALIDE_FUNCTION_ATTRS;
int recfilter_ref(buffer_t *_b_input_buffer, buffer_t *_b_blury_buffer) HALIDE_FUNCTION_ATTRS;
int recfilter_ref_argv(void **args) HALIDE_FUNCTION_ATTRS;
// Result is never null and points to constant static data
const struct halide_filter_metadata_t *recfilter_coli_metadata() HALIDE_FUNCTION_ATTRS;
const struct halide_filter_metadata_t *recfilter_ref_metadata() HALIDE_FUNCTION_ATTRS;

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
