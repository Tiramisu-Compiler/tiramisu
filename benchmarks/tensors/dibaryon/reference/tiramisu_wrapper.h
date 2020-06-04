#ifndef TIRAMISU_DIBARYON_WRAPPER_h
#define TIRAMISU_DIBARYON_WRAPPER_h

#ifdef __cplusplus
extern "C" {
#endif

int index_2d(int a, int b, int length2);
int index_3d(int a, int b, int c, int length2, int length3);
int index_4d(int a, int b, int c, int d, int length2, int length3, int length4);
int prop_index(int q, int t, int c1, int s1, int c2, int s2, int y, int x, int Nc, int Ns, int Vsrc, int Vsnk, int Nt);


void tiramisu_make_nucleon_2pt(double* C_re,
    double* C_im,
    const double* B1_prop_re, 
    const double* B1_prop_im, 
    const int *src_color_weights_r1,
    const int *src_spin_weights_r1,
    const double *src_weights_r1,
    const int *src_color_weights_r2,
    const int *src_spin_weights_r2,
    const double *src_weights_r2,
    const double* src_psi_B1_re, 
    const double* src_psi_B1_im, 
    const double* snk_psi_B1_re, 
    const double* snk_psi_B1_im,
    const int Nc,
    const int Ns,
    const int Vsrc,
    const int Vsnk,
    const int Lt,
    const int Nw,
    const int Nq,
    const int NsrcHex,
    const int NsnkHex);

void tiramisu_make_two_nucleon_2pt(double* C_re,
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
    const int *src_color_weights_A1,
    const int *src_spin_weights_A1,
    const double *src_weights_A1,
    const int *src_color_weights_T1_r1,
    const int *src_spin_weights_T1_r1,
    const double *src_weights_T1_r1,
    const int *src_color_weights_T1_r2,
    const int *src_spin_weights_T1_r2,
    const double *src_weights_T1_r2,
    const int *src_color_weights_T1_r3,
    const int *src_spin_weights_T1_r3,
    const double *src_weights_T1_r3,
    const int* perms, 
    const int* sigs, 
    const double* src_psi_B1_re, 
    const double* src_psi_B1_im, 
    const double* src_psi_B2_re, 
    const double* src_psi_B2_im, 
    const double* snk_psi_re,
    const double* snk_psi_im, 
    const double* snk_psi_B1_re, 
    const double* snk_psi_B1_im, 
    const double* snk_psi_B2_re, 
    const double* snk_psi_B2_im, 
    const double* hex_src_psi_re, 
    const double* hex_src_psi_im, 
    const double* hex_snk_psi_re, 
    const double* hex_snk_psi_im,
    const int Nc,
    const int Ns,
    const int Vsrc,
    const int Vsnk,
    const int Lt,
    const int Nw,
    const int Nq,
    const int Nsrc,
    const int Nsnk,
    const int NsrcHex,
    const int NsnkHex,
    const int Nperms);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
