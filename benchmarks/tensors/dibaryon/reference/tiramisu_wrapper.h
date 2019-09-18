#ifndef TIRAMISU_DIBARYON_WRAPPER_h
#define TIRAMISU_DIBARYON_WRAPPER_h

#ifdef __cplusplus
extern "C" {
#endif

void tiramisu_wrapper_make_local_single_double_block(double *t_B1_Blocal_r1_re, double *t_B1_Blocal_r1_im, double *t_B1_Bsingle_r1_re, double *t_B1_Bsingle_r1_im, double *t_B1_Bdouble_r1_re, double *t_B1_Bdouble_r1_im, const double *B1_prop_re, const double *B1_prop_im, const int *src_color_weights_r1, const int *src_spin_weights_r1, const double *src_weights_r1, const double *src_psi_B1_re, const double *src_psi_B1_im, const int Nc, const int Ns, const int Vsrc, const int Vsnk, const int Nt, const int Nw, const int Nq, const int Nsrc);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
