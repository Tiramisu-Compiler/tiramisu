#ifndef MARK_29681028_3ABA_81FF_EBA1_781947E184FF
#define MARK_29681028_3ABA_81FF_EBA1_781947E184FF

#if ((!defined USE_REFERENCE) && (!defined USE_TIRAMISU))
    #define USE_REFERENCE 1
    #define USE_TIRAMISU 0
#endif

int index_2d(int a, int b, int length2);
int index_3d(int a, int b, int c, int length2, int length3);
int index_4d(int a, int b, int c, int d, int length2, int length3, int length4);
int prop_index(int q, int t, int c1, int s1, int c2, int s2, int y, int x, int Nc, int Ns, int Vsrc, int Vsnk, int Nt);

void make_two_nucleon_2pt(double* C_re,
    double* C_im,
    const double* B1_prop_re, 
    const double* B1_prop_im, 
    const double* B2_prop_re, 
    const double* B2_prop_im, 
    const int* src_color_weights_r1, 
    const int* src_spin_weights_r1, 
    const double* src_weights_r1, 
    const int* src_color_weights_r2, 
    const int* src_spin_weights_r2, 
    const double* src_weights_r2, 
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
    const int space_symmetric,
    const int snk_entangled,
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

void make_nucleon_2pt(double* C_re,
    double* C_im,
    const double* prop_re, 
    const double* prop_im, 
    const int* src_color_weights_r1, 
    const int* src_spin_weights_r1, 
    const double* src_weights_r1, 
    const int* src_color_weights_r2, 
    const int* src_spin_weights_r2, 
    const double* src_weights_r2, 
    const double* src_psi_re, 
    const double* src_psi_im, 
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
    const int Nsnk);

/* Erorr threshold. If the difference between the results of Tiramisu and the reference code is bigger than this value,
 we signal an error. */
#define ERROR_THRESH 1e-12

#endif /* !defined(MARK_29681028_3ABA_81FF_EBA1_781947E184FF) */
