#ifndef TIRAMISU_DIBARYON_GENERATED_WRAPPER_h
#define TIRAMISU_DIBARYON_GENERATED_WRAPPER_h

#include <tiramisu/utils.h>

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

int tiramisu_make_dibaryon_correlator(struct halide_buffer_t *_buf_C_r_buffer, struct halide_buffer_t *_buf_C_i_buffer, struct halide_buffer_t *_B1_Blocal_re_b2_buffer, struct halide_buffer_t *_B1_Blocal_im_b3_buffer, struct halide_buffer_t *_B1_Bsingle_re_b4_buffer, struct halide_buffer_t *_B1_Bsingle_im_b5_buffer, struct halide_buffer_t *_B1_Bdouble_re_b6_buffer, struct halide_buffer_t *_B1_Bdouble_im_b7_buffer, struct halide_buffer_t *_B2_Blocal_re_b8_buffer, struct halide_buffer_t *_B2_Blocal_im_b9_buffer, struct halide_buffer_t *_B2_Bsingle_re_b10_buffer, struct halide_buffer_t *_B2_Bsingle_im_b11_buffer, struct halide_buffer_t *_B2_Bdouble_re_b12_buffer, struct halide_buffer_t *_B2_Bdouble_im_b13_buffer, struct halide_buffer_t *_perms_b14_buffer, struct halide_buffer_t *_sigs_b15_buffer, struct halide_buffer_t *_overall_weight_b16_buffer, struct halide_buffer_t *_snk_color_weights_b17_buffer, struct halide_buffer_t *_snk_spin_weights_b18_buffer, struct halide_buffer_t *_snk_weights_b19_buffer, struct halide_buffer_t *_buf_snk_psi_re_buffer, struct halide_buffer_t *_buf_snk_psi_im_buffer,

		struct halide_buffer_t * term_re,
	        struct halide_buffer_t * term_im,
		struct halide_buffer_t * snk_1,
		struct halide_buffer_t * snk_1_b,
		struct halide_buffer_t * snk_1_nq
		);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
