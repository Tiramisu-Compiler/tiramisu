#ifndef TIRAMISU_DIBARYON_WRAPPER_h
#define TIRAMISU_DIBARYON_WRAPPER_h

#ifdef __cplusplus
extern "C" {
#endif


void tiramisu_wrapper_make_fused_blocks_dibaryon_correlator(double* C_re,
    double* C_im,
    const double* B1_prop_re, 
    const double* B1_prop_im, 
    const double* B2_prop_re, 
    const double* B2_prop_im, 
	 const int *src_color_weights_r1,
	 const int *src_spin_weights_r1,
	 const double *src_weights_r1,
	 const int *src_color_weights_r2,
	 const int *src_spin_weights_r2,
	 const double *src_weights_r2,
    const int* perms, 
    const int* sigs, 
    const double* src_psi_B1_re, 
    const double* src_psi_B1_im, 
    const double* src_psi_B2_re, 
    const double* src_psi_B2_im, 
    const double* snk_psi_re,
    const double* snk_psi_im,
    const int Nc,
    const int Ns,
    const int Vsrc,
    const int Vsnk,
    const int Nt,
    const int Nw,
    const int Nq,
    const int Nsrc,
    const int Nsnk,
    const int NsrcHex,
    const int NsnkHex,
    const int Nperms);

/*
void tiramisu_wrapper_make_local_single_double_block(int r1, double *t_B1_Blocal_r1_re, double *t_B1_Blocal_r1_im, double *t_B1_Bsingle_r1_re, double *t_B1_Bsingle_r1_im, double *t_B1_Bdouble_r1_re, double *t_B1_Bdouble_r1_im, const double *B1_prop_re, const double *B1_prop_im, const int *src_color_weights_r1, const int *src_spin_weights_r1, const double *src_weights_r1, const double *src_psi_B1_re, const double *src_psi_B1_im, const int Nc, const int Ns, const int Vsrc, const int Vsnk, const int Nt, const int Nw, const int Nq, const int Nsrc);

void tiramisu_wrapper_make_dibaryon_correlator(double* t_C_re,
    double* t_C_im,
    double* t_B1_Blocal_re,
    double* t_B1_Blocal_im,
    double* t_B1_Bsingle_re,
    double* t_B1_Bsingle_im,
    double* t_B1_Bdouble_re,
    double* t_B1_Bdouble_im,
    double* t_B2_Blocal_re,
    double* t_B2_Blocal_im,
    double* t_B2_Bsingle_re,
    double* t_B2_Bsingle_im,
    double* t_B2_Bdouble_re,
    double* t_B2_Bdouble_im,
    const int* t_perms,
    const int* t_sigs,
    const double *t_overall_weight,
    const int* t_snk_color_weights,
    const int* t_snk_spin_weights,
    const double* t_snk_weights,
    const double* t_snk_psi_re,
    const double* t_snk_psi_im,
    const int Nc,
    const int Ns,
    const int Vsrc,
    const int Vsnk,
    const int Nt,
    const int Nw,
    const int Nq,
    const int Nsrc,
    const int Nsnk,
    const int Nperms);
    */

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
