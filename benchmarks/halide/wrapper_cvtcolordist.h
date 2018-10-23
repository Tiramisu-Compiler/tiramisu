#ifndef HALIDE__build___wrapper_cvtcolordist_o_h
#define HALIDE__build___wrapper_cvtcolordist_o_h

#include <tiramisu/utils.h>

#ifndef NODES
#define NODES 4
#endif


#ifdef __cplusplus
extern "C" {
#endif

int cvtcolordist_tiramisu(halide_buffer_t *_b_input_buffer, halide_buffer_t *_b_blury_buffer);
int cvtcolordist_tiramisu_argv(void **args);
int cvtcolordist_ref(halide_buffer_t *_b_input_buffer, halide_buffer_t *_b_blury_buffer);
int cvtcolordist_ref_argv(void **args);
// Result is never null and points to constant static data
const struct halide_filter_metadata_t *cvtcolordist_tiramisu_metadata();
const struct halide_filter_metadata_t *cvtcolordist_ref_metadata();

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
