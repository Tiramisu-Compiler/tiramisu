#ifndef TIRAMISU_test_h
#define TIRAMISU_test_h

#include "../reference/qblocks_2pt_parameters.h"

#define Nq P_Nq
#define Nc P_Nc
#define Ns P_Ns
#define Nw P_Nw
 #define Nw2 P_Nw2
//#define Nw2 Nw*Nw
#define Nw2Hex P_Nw2Hex
#define Nperms P_Nperms
#define Lt P_Nt
#define Vsrc P_Vsrc
#define Vsnk P_Vsnk
#define Nsrc P_Nsrc
#define Nsnk P_Nsnk
#define NsrcHex P_NsrcHex
#define NsnkHex P_NsnkHex
#define mq P_mq
#define Nb P_Nb
#define B2Nrows P_B2Nrows
#define B1Nrows P_B1Nrows
#define NEntangled P_NEntangled
#define sites_per_rank P_sites_per_rank
#define src_sites_per_rank P_src_sites_per_rank

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

// Define these values for each new test
#define TEST_NAME_STR       "tiramisu_make_fused_dibaryon_blocks_correaltor"

#include <tiramisu/utils.h>

#ifdef __cplusplus
extern "C" {
#endif

int tiramisu_make_fused_dibaryon_blocks_correlator(
			    halide_buffer_t *,
			    halide_buffer_t *,
			    halide_buffer_t *,
			    halide_buffer_t *,
			    halide_buffer_t *,
			    halide_buffer_t *,
			    halide_buffer_t *,
			    halide_buffer_t *,
			    halide_buffer_t *,
			    halide_buffer_t *,
			    halide_buffer_t *,
			    halide_buffer_t *,
			    halide_buffer_t *,
			    halide_buffer_t *,
			    halide_buffer_t *,
			    halide_buffer_t *,
			    halide_buffer_t *,
			    halide_buffer_t *,
			    halide_buffer_t *,
			    halide_buffer_t *,
			    halide_buffer_t *,
			    halide_buffer_t *,
			    halide_buffer_t *,
			    halide_buffer_t *,
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
