#ifndef MARK_29681028_3ABA_81FF_EBA1_781947E184FF
#define MARK_29681028_3ABA_81FF_EBA1_781947E184FF

//fixed parameters 

#define Nc 3
#define Ns 2
#define Nq 3

#define Nw 9
#define Nperms 9

#define Vsrc 1
#define Vsnk 1
#define Nt 1

#define Nsrc 4
#define Nsnk 4
#define NsrcHex 1
#define NsnkHex 1

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
    const int snk_entangled);

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
    const double* snk_psi_im);

void make_local_block(double* Blocal_re, 
    double* Blocal_im, 
    const double* prop_re,
    const double* prop_im, 
    const int* color_weights, 
    const int* spin_weights, 
    const double* weights, 
    const double* psi_re, 
    const double* psi_im);

void make_local_snk_block(double* Blocal_re, 
    double* Blocal_im, 
    const double* prop_re,
    const double* prop_im, 
    const int* color_weights, 
    const int* spin_weights, 
    const double* weights, 
    const double* psi_re, 
    const double* psi_im);

void make_single_block(double* Bsingle_re, 
    double* Bsingle_im, 
    const double* prop_re,
    const double* prop_im, 
    const int* color_weights, 
    const int* spin_weights, 
    const double* weights, 
    const double* psi_re, 
    const double* psi_im);

void make_double_block(double* Bdouble_re, 
    double* Bdouble_im, 
    const double* prop_re,
    const double* prop_im, 
    const int* color_weights, 
    const int* spin_weights, 
    const double* weights, 
    const double* psi_re, 
    const double* psi_im);

void make_dibaryon_correlator(double* C_re,
    double* C_im,
    const double* B1_Blocal_re, 
    const double* B1_Blocal_im, 
    const double* B1_Bsingle_re, 
    const double* B1_Bsingle_im, 
    const double* B1_Bdouble_re, 
    const double* B1_Bdouble_im, 
    const double* B2_Blocal_re, 
    const double* B2_Blocal_im, 
    const double* B2_Bsingle_re, 
    const double* B2_Bsingle_im, 
    const double* B2_Bdouble_re, 
    const double* B2_Bdouble_im, 
    const int* perms, 
    const int* sigs, 
    const double overall_weight,
    const int* snk_color_weights, 
    const int* snk_spin_weights, 
    const double* snk_weights, 
    const double* snk_psi_re, 
    const double* snk_psi_im);

void make_dibaryon_hex_correlator(double* C_re,
    double* C_im,
    const double* B1_Blocal_re, 
    const double* B1_Blocal_im, 
    const double* B2_Blocal_re, 
    const double* B2_Blocal_im, 
    const int* perms, 
    const int* sigs, 
    const double overall_weight,
    const int* snk_color_weights, 
    const int* snk_spin_weights, 
    const double* snk_weights, 
    const double* hex_snk_psi_re,
    const double* hex_snk_psi_im);

void make_hex_dibaryon_correlator(double* C_re,
    double* C_im,
    const double* B1_Blocal_re, 
    const double* B1_Blocal_im, 
    const double* B2_Blocal_re, 
    const double* B2_Blocal_im, 
    const int* perms, 
    const int* sigs, 
    const double overall_weight,
    const int* src_color_weights, 
    const int* src_spin_weights, 
    const double* src_weights, 
    const double* hex_src_psi_re,
    const double* hex_src_psi_im);

void make_hex_correlator(double* C_re,
    double* C_im,
    const double* B1_props_re, 
    const double* B1_props_im, 
    const double* B2_props_re, 
    const double* B2_props_im, 
    const int* perms, 
    const int* sigs, 
    const int* B1_color_weights, 
    const int* B1_spin_weights, 
    const double* B1_weights, 
    const int* B2_color_weights, 
    const int* B2_spin_weights, 
    const double* B2_weights, 
    const double overall_weight,
    const int* color_weights, 
    const int* spin_weights, 
    const double* weights, 
    const double* hex_src_psi_re,
    const double* hex_src_psi_im,
    const double* hex_snk_psi_re,
    const double* hex_snk_psi_im);

int prop_index(int q, int t, int c1, int s1, int c2, int s2, int y, int x);
int Q_index(int t, int c1, int s1, int c2, int s2, int x1, int c3, int s3, int y);
int Blocal_index(int t, int c1, int s1, int c2, int s2, int x, int c3, int s3, int m);
int snk_Blocal_index(int t, int c1, int s1, int c2, int s2, int y, int c3, int s3, int n);
int Bsingle_index(int t, int c1, int s1, int c2, int s2, int x1, int c3, int s3, int x2, int m);
int Bdouble_index(int t, int c1, int s1, int c2, int s2, int x1, int c3, int s3, int x2, int m);
int src_weight_index(int nw, int nq);
int snk_weight_index(int nb, int nw, int nq);
int src_psi_index(int y, int m);
int snk_one_psi_index(int x, int n);
int snk_psi_index(int x1, int x2, int n);
int hex_src_psi_index(int y, int m);
int hex_snk_psi_index(int x, int n);
int perm_index(int n, int q);
int one_correlator_index(int m, int n, int t);
int one_hex_correlator_index(int m, int n, int t);
int correlator_index(int r, int m, int n, int t);
int B1_correlator_index(int r, int m, int n, int t);

#endif /* !defined(MARK_29681028_3ABA_81FF_EBA1_781947E184FF) */
