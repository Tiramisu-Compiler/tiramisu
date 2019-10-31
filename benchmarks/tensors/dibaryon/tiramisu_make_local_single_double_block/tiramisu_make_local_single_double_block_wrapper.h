#ifndef TIRAMISU_test_h
#define TIRAMISU_test_h

#include "../reference/qblocks_2pt_parameters.h"

#define SMALL_BARYON_DATA_SET 0
#define LARGE_BARYON_DATA_SET 0
#define USE_GLOBAL_PARAMS 1

#define R1 0

#if SMALL_BARYON_DATA_SET

#define Nq 3
#define Nc 3
#define Ns 2
#define Nw 9
#define twoNw 81
#define Nperms 36
#define Lt 2
#define Vsrc 2
#define Vsnk 4
#define Nsrc 2
#define Nsnk 2
#define mq 1.0

#elif LARGE_BARYON_DATA_SET

#define Nq 3
#define Nc 3
#define Ns 2
#define Nw 9
#define twoNw 81
#define Nperms 36
#define Lt 4 // 1..32
#define Vsrc 16 //64 //8, 64, 512
#define Vsnk 16 //64 //8, 64, 512
#define Nsrc 6
#define Nsnk 6
#define mq 1.0

#elif USE_GLOBAL_PARAMS

#define Nq P_Nq
#define Nc P_Nc
#define Ns P_Ns
#define Nw P_Nw
#define twoNw Nw*Nw
#define Nperms P_Nperms
#define Lt P_Nt
#define Vsrc P_Vsrc
#define Vsnk P_Vsnk
#define Nsrc P_Nsrc
#define Nsnk P_Nsnk
#define mq P_mq

#endif

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

// Define these values for each new test
#define TEST_NAME_STR       "tiramisu_make_local_single_double_block"

#include <tiramisu/utils.h>

static int src_color_weights_r1_P[Nw][Nq] = { {0,1,2}, {0,2,1}, {1,0,2} ,{0,1,2}, {0,2,1}, {1,0,2}, {1,2,0}, {2,1,0}, {2,0,1} };
static int src_spin_weights_r1_P[Nw][Nq] = { {0,1,0}, {0,1,0}, {0,1,0}, {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0} };
static double src_weights_r1_P[Nw] = {-2/ sqrt(2), 2/sqrt(2), 2/sqrt(2), 1/sqrt(2), -1/sqrt(2), -1/sqrt(2), 1/sqrt(2), -1/sqrt(2), 1/sqrt(2)};

static int src_color_weights_r2_P[Nw][Nq] = { {0,1,2}, {0,2,1}, {1,0,2} ,{1,2,0}, {2,1,0}, {2,0,1}, {0,1,2}, {0,2,1}, {1,0,2} };
static int src_spin_weights_r2_P[Nw][Nq] = { {0,1,1}, {0,1,1}, {0,1,1}, {0,1,1}, {0,1,1}, {0,1,1}, {1,0,1}, {1,0,1}, {1,0,1} };
static double src_weights_r2_P[Nw] = {1/ sqrt(2), -1/sqrt(2), -1/sqrt(2), 1/sqrt(2), -1/sqrt(2), 1/sqrt(2), -2/sqrt(2), 2/sqrt(2), 2/sqrt(2)};

#ifdef __cplusplus
extern "C" {
#endif

int tiramisu_make_local_single_double_block_r1(halide_buffer_t *,
			    halide_buffer_t *,
			    halide_buffer_t *,
			    halide_buffer_t *,
			    halide_buffer_t *,
			    halide_buffer_t *,
			    halide_buffer_t *,
			    halide_buffer_t *,
	   		    halide_buffer_t *,
			    halide_buffer_t *);

int tiramisu_make_local_single_double_block_r2(halide_buffer_t *,
			    halide_buffer_t *,
			    halide_buffer_t *,
			    halide_buffer_t *,
			    halide_buffer_t *,
			    halide_buffer_t *,
			    halide_buffer_t *,
			    halide_buffer_t *,
	   		    halide_buffer_t *,
			    halide_buffer_t *);

int tiramisu_generated_code_argv(void **args);

extern const struct halide_filter_metadata_t halide_pipeline_aot_metadata;
#ifdef __cplusplus
}  // extern "C"
#endif
#endif
