#ifndef HALIDE__build___wrapper_cvtcolordist_o_h
#define HALIDE__build___wrapper_cvtcolordist_o_h

#include <tiramisu/utils.h>

#ifndef NODES
#define NODES 10
#endif


#ifdef __cplusplus
extern "C" {
#endif

int cvtcolorautodist_tiramisu(halide_buffer_t *_b_input_buffer, halide_buffer_t *_b_blury_buffer);
int cvtcolorautodist_tiramisu_argv(void **args);
int cvtcolorautodist_ref(halide_buffer_t *_b_input_buffer, halide_buffer_t *_b_blury_buffer);
int cvtcolorautodist_ref_argv(void **args);
// Result is never null and points to constant static data
const struct halide_filter_metadata_t *cvtcolorautodist_tiramisu_metadata();
const struct halide_filter_metadata_t *cvtcolorautodist_ref_metadata();

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
