#ifndef HALIDE__build___wrapper_warpdist_o_h
#define HALIDE__build___wrapper_warpdist_o_h

#include <tiramisu/utils.h>



#ifdef __cplusplus
extern "C" {
#endif

#define _N0 1500
#define _N1 1500
#define NODES 10

int warpdist_tiramisu(halide_buffer_t *_b_SIZES, halide_buffer_t *_b_input_buffer, halide_buffer_t *_b_blury_buffer);
int warpdist_tiramisu_argv(void **args);
int warpdist_ref(halide_buffer_t *_b_input_buffer, halide_buffer_t *_b_blury_buffer);
int warpdist_ref_argv(void **args);
// Result is never null and points to constant static data
const struct halide_filter_metadata_t *warpdist_tiramisu_metadata();
const struct halide_filter_metadata_t *warpdist_ref_metadata();

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
