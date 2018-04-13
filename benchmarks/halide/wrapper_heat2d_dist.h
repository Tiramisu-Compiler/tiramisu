#ifndef HALIDE__build___wrapper_heat2d_dist_o_h
#define HALIDE__build___wrapper_heat2d_dist_o_h

#include <tiramisu/utils.h>

#define N_TESTS 10
#define CHECK_CORRECTNESS false

#define USE_MPI

#define N ((int64_t)49994)
/*percentage, ex: 10 = 100%, 1 = 10%*/
#define CPU_SPLIT ((int)10)
/*percentage*/
#define GPU_SPLIT (100-CPU_SPLIT) 
#define NUM_CPU_RANKS ((int)2)
#define NUM_GPU_RANKS ((int)0)

#ifdef __cplusplus
extern "C" {
#endif

int heat2d_dist_tiramisu(halide_buffer_t *_b_input_buffer, halide_buffer_t *_b_output_buffer);
int heat2d_dist_tiramisu_argv(void **args);
int heat2d_dist_ref(halide_buffer_t *_b_input_buffer, halide_buffer_t *_b_output_buffer);
int heat2d_dist_ref_argv(void **args);
// Result is never null and points to constant static data
const struct halide_filter_metadata_t *heat2d_dist_tiramisu_metadata();
const struct halide_filter_metadata_t *heat2d_dist_ref_metadata();

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
