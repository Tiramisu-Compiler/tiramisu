#ifndef HALIDE__build___wrapper_blurxyautodist_o_h
#define HALIDE__build___wrapper_blurxyautodist_o_h

#include <tiramisu/utils.h>
#define _ROWS 100
#define _COLS 10

#ifndef NODES
#define NODES 10
#endif


#ifdef __cplusplus
extern "C" {
#endif

int blurxyautodist_tiramisu(halide_buffer_t *_b_input_buffer, halide_buffer_t *_b_blury_buffer);
int blurxyautodist_tiramisu_argv(void **args);
int blurxyautodist_ref(halide_buffer_t *_b_input_buffer, halide_buffer_t *_b_blury_buffer);
int blurxyautodist_ref_argv(void **args);

// Result is never null and points to constant static data
const struct halide_filter_metadata_t *blurxyautodist_tiramisu_metadata();
const struct halide_filter_metadata_t *blurxyautodist_ref_metadata();

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
