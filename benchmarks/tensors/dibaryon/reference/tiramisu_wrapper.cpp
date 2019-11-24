
/*#if USE_REFERENCE

void tiramisu_wrapper_make_local_single_double_block(int r1,
	double *t_B1_Blocal_r1_re,
	double *t_B1_Blocal_r1_im,
	double *t_B1_Bsingle_r1_re,
	double *t_B1_Bsingle_r1_im,
	double *t_B1_Bdouble_r1_re,
	double *t_B1_Bdouble_r1_im,
	const double *B1_prop_re,
	const double *B1_prop_im,
	const int *src_color_weights_r1,
	const int *src_spin_weights_r1,
	const double *src_weights_r1,
	const double *src_psi_B1_re,
	const double *src_psi_B1_im,
	const int Nc, const int Ns,
	const int Vsrc,
	const int Vsnk,
	const int Nt,
	const int Nw,
	const int Nq,
	const int Nsrc);

void tiramisu_wrapper_make_dibaryon_correlator(double* C_re,
    double* C_im,
    double* B1_Blocal_re,
    double* B1_Blocal_im,
    double* B1_Bsingle_re,
    double* B1_Bsingle_im,
    double* B1_Bdouble_re,
    double* B1_Bdouble_im,
    double* B2_Blocal_re,
    double* B2_Blocal_im,
    double* B2_Bsingle_re,
    double* B2_Bsingle_im,
    double* B2_Bdouble_re,
    double* B2_Bdouble_im,
    int* perms,
    int* sigs,
    double *overall_weight,
    int* snk_color_weights,
    int* snk_spin_weights,
    double* snk_weights,
    double* snk_psi_re,
    double* snk_psi_im,
    const int Nc,
    const int Ns,
    const int Vsrc,
    const int Vsnk,
    const int Nt,
    const int Nw,
    const int Nq,
    const int Nsrc,
    const int Nsnk,
    const int Nperms)
 */

////////////////////////////////////////////////////////////
#include <cstdlib>
#include <iostream>
#include <complex>

#include "Halide.h"
#include <tiramisu/utils.h>

#include "tiramisu_generated_wrapper.h"
#include <stdio.h>
#include "qblocks_2pt.h"

#ifdef __cplusplus
extern "C" {
#endif

int index_2d_cpp(int a, int b, int length2) {
   return b +length2*( a );
}
int index_3d_cpp(int a, int b, int c, int length2, int length3) {
   return c +length3*( b +length2*( a ));
}
int index_4d_cpp(int a, int b, int c, int d, int length2, int length3, int length4) {
   return d +length4*( c +length3*( b +length2*( a )));
}

   /*
void tiramisu_wrapper_make_local_single_double_block(int r1, double *t_B1_Blocal_r1_re, double *t_B1_Blocal_r1_im, double *t_B1_Bsingle_r1_re, double *t_B1_Bsingle_r1_im, double *t_B1_Bdouble_r1_re, double *t_B1_Bdouble_r1_im, const double *B1_prop_re, const double *B1_prop_im, const int *src_color_weights_r1, const int *src_spin_weights_r1, const double *src_weights_r1, const double *src_psi_B1_re, const double *src_psi_B1_im, const int Nc, const int Ns, const int Vsrc, const int Vsnk, const int Nt, const int Nw, const int Nq, const int Nsrc)
{
    // Blocal
    // B1_Blocal_r1_r: tiramisu real part of Blocal.
    // B1_Blocal_r1_i: tiramisu imaginary part of Blocal.
    Halide::Buffer<double> B1_Blocal_r1_r(t_B1_Blocal_r1_re, {Ns, Nc, Nsrc, Vsnk, Ns, Nc, Ns, Nc, Nt});
    Halide::Buffer<double> B1_Blocal_r1_i(t_B1_Blocal_r1_im, {Ns, Nc, Nsrc, Vsnk, Ns, Nc, Ns, Nc, Nt});

    // prop
    Halide::Buffer<double> B1_prop_r((double *)B1_prop_re, {Vsrc, Vsnk, Ns, Nc, Ns, Nc, Nt, Nq});
    Halide::Buffer<double> B1_prop_i((double *)B1_prop_im, {Vsrc, Vsnk, Ns, Nc, Ns, Nc, Nt, Nq});

    // psi
    Halide::Buffer<double> psi_r((double *)src_psi_B1_re, {Vsrc, Nsrc});
    Halide::Buffer<double> psi_i((double *)src_psi_B1_im, {Vsrc, Nsrc});

    Halide::Buffer<int> color_weights_t((int *)src_color_weights_r1, {Nq, Nw});
    Halide::Buffer<int> spin_weights_t((int *)src_spin_weights_r1, {Nq, Nw});
    Halide::Buffer<double> weights_t((double *)src_weights_r1, Nw);

    Halide::Buffer<double> B1_Bsingle_r1_r(t_B1_Bsingle_r1_re, {Vsnk, Ns, Nc, Nsrc, Vsnk, Ns, Nc, Ns, Nc, Nt});
    Halide::Buffer<double> B1_Bsingle_r1_i(t_B1_Bsingle_r1_im, {Vsnk, Ns, Nc, Nsrc, Vsnk, Ns, Nc, Ns, Nc, Nt});

    Halide::Buffer<double> B1_Bdouble_r1_r(t_B1_Bdouble_r1_re, {Vsnk, Ns, Nc, Nsrc, Vsnk, Ns, Nc, Ns, Nc, Nt});
    Halide::Buffer<double> B1_Bdouble_r1_i(t_B1_Bdouble_r1_im, {Vsnk, Ns, Nc, Nsrc, Vsnk, Ns, Nc, Ns, Nc, Nt});

    if (r1)
        tiramisu_make_local_single_double_block_r1(B1_Blocal_r1_r.raw_buffer(),
				    B1_Blocal_r1_i.raw_buffer(),
				    B1_prop_r.raw_buffer(),
				    B1_prop_i.raw_buffer(),
				    psi_r.raw_buffer(),
				    psi_i.raw_buffer(),
				    B1_Bsingle_r1_r.raw_buffer(),
				    B1_Bsingle_r1_i.raw_buffer(),
				    B1_Bdouble_r1_r.raw_buffer(),
				    B1_Bdouble_r1_i.raw_buffer());
    else
        tiramisu_make_local_single_double_block_r2(B1_Blocal_r1_r.raw_buffer(),
				    B1_Blocal_r1_i.raw_buffer(),
				    B1_prop_r.raw_buffer(),
				    B1_prop_i.raw_buffer(),
				    psi_r.raw_buffer(),
				    psi_i.raw_buffer(),
				    B1_Bsingle_r1_r.raw_buffer(),
				    B1_Bsingle_r1_i.raw_buffer(),
				    B1_Bdouble_r1_r.raw_buffer(),
				    B1_Bdouble_r1_i.raw_buffer());
} */

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
    const int Nperms)
{
    int Nb = 2;
    int Nr = 6;
    int Nw2 = Nw*Nw;

    // prop
    Halide::Buffer<double> b_B1_prop_r((double *)B1_prop_re, {Vsrc, Vsnk, Ns, Nc, Ns, Nc, Nt, Nq});
    Halide::Buffer<double> b_B1_prop_i((double *)B1_prop_im, {Vsrc, Vsnk, Ns, Nc, Ns, Nc, Nt, Nq});
    Halide::Buffer<double> b_B2_prop_r((double *)B2_prop_re, {Vsrc, Vsnk, Ns, Nc, Ns, Nc, Nt, Nq});
    Halide::Buffer<double> b_B2_prop_i((double *)B2_prop_im, {Vsrc, Vsnk, Ns, Nc, Ns, Nc, Nt, Nq});

    // psi
    Halide::Buffer<double> b_B1_src_psi_r((double *)src_psi_B1_re, {Vsrc, Nsrc});
    Halide::Buffer<double> b_B1_src_psi_i((double *)src_psi_B1_im, {Vsrc, Nsrc});
    Halide::Buffer<double> b_B2_src_psi_r((double *)src_psi_B1_re, {Vsrc, Nsrc});
    Halide::Buffer<double> b_B2_src_psi_i((double *)src_psi_B1_im, {Vsrc, Nsrc});
    Halide::Buffer<double> b_snk_psi_r((double *)snk_psi_re, {Nsnk, Vsnk, Vsnk});
    Halide::Buffer<double> b_snk_psi_i((double *)snk_psi_im, {Nsnk, Vsnk, Vsnk});

   // Weights
   Halide::Buffer<int> b_snk_b(2, Nq, Nperms, "snk_b");
   Halide::Buffer<int> b_snk_color_weights(2, Nq, Nw2, Nperms, Nr, "snk_color_weights");
   Halide::Buffer<int> b_snk_spin_weights(2, Nq, Nw2, Nperms, Nr, "snk_spin_weights");
   Halide::Buffer<double> b_snk_weights(Nw2, Nr, "snk_weights");

   Halide::Buffer<int> b_sigs((int *)sigs, Nperms);

   Halide::Buffer<int> b_snk_blocks(2, Nr, "snk_blocks");
   b_snk_blocks(0,0) = 1;
   b_snk_blocks(1,0) = 1;
   b_snk_blocks(0,1) = 2;
   b_snk_blocks(1,1) = 2;
   b_snk_blocks(0,2) = 1;
   b_snk_blocks(1,2) = 2;
   b_snk_blocks(0,3) = 2;
   b_snk_blocks(1,3) = 1;
   b_snk_blocks(0,4) = 1;
   b_snk_blocks(1,4) = 2;
   b_snk_blocks(0,5) = 2;
   b_snk_blocks(1,5) = 1;

   int* snk_color_weights_r1 = (int *) malloc(2 * Nw2 * Nq * sizeof (int));
   int* snk_color_weights_r2_1 = (int *) malloc(2 * Nw2 * Nq * sizeof (int));
   int* snk_color_weights_r2_2 = (int *) malloc(2 * Nw2 * Nq * sizeof (int));
   int* snk_color_weights_r3 = (int *) malloc(2 * Nw2 * Nq * sizeof (int));
   int* snk_spin_weights_r1 = (int *) malloc(2 * Nw2 * Nq * sizeof (int));
   int* snk_spin_weights_r2_1 = (int *) malloc(2 * Nw2 * Nq * sizeof (int));
   int* snk_spin_weights_r2_2 = (int *) malloc(2 * Nw2 * Nq * sizeof (int));
   int* snk_spin_weights_r3 = (int *) malloc(2 * Nw2 * Nq * sizeof (int));
   for (int nB1=0; nB1<Nw; nB1++) {
      for (int nB2=0; nB2<Nw; nB2++) {
         b_snk_weights(nB1+Nw*nB2, 0) = src_weights_r1[nB1]*src_weights_r1[nB2];
         b_snk_weights(nB1+Nw*nB2, 1) = src_weights_r2[nB1]*src_weights_r2[nB2];
         b_snk_weights(nB1+Nw*nB2, 2) = 1.0/sqrt(2) * src_weights_r1[nB1]*src_weights_r2[nB2];
         b_snk_weights(nB1+Nw*nB2, 3) = 1.0/sqrt(2) * src_weights_r1[nB1]*src_weights_r2[nB2];
         b_snk_weights(nB1+Nw*nB2, 4) = 1.0/sqrt(2) * src_weights_r2[nB1]*src_weights_r1[nB2];
         b_snk_weights(nB1+Nw*nB2, 5) = 1.0/sqrt(2) * src_weights_r2[nB1]*src_weights_r1[nB2];
         for (int nq=0; nq<Nq; nq++) {
            // T1g_r1
            snk_color_weights_r1[index_3d_cpp(0,nB1+Nw*nB2,nq ,Nw2,Nq)] = src_color_weights_r1[index_2d_cpp(nB1,nq ,Nq)];
            snk_spin_weights_r1[index_3d_cpp(0,nB1+Nw*nB2,nq ,Nw2,Nq)] = src_spin_weights_r1[index_2d_cpp(nB1,nq ,Nq)];
            snk_color_weights_r1[index_3d_cpp(1,nB1+Nw*nB2,nq ,Nw2,Nq)] = src_color_weights_r1[index_2d_cpp(nB2,nq ,Nq)];
            snk_spin_weights_r1[index_3d_cpp(1,nB1+Nw*nB2,nq ,Nw2,Nq)] = src_spin_weights_r1[index_2d_cpp(nB2,nq ,Nq)];
            // T1g_r2
            snk_color_weights_r2_1[index_3d_cpp(0,nB1+Nw*nB2,nq ,Nw2,Nq)] = src_color_weights_r1[index_2d_cpp(nB1,nq ,Nq)];
            snk_spin_weights_r2_1[index_3d_cpp(0,nB1+Nw*nB2,nq ,Nw2,Nq)] = src_spin_weights_r1[index_2d_cpp(nB1,nq ,Nq)];
            snk_color_weights_r2_1[index_3d_cpp(1,nB1+Nw*nB2,nq ,Nw2,Nq)] = src_color_weights_r2[index_2d_cpp(nB2,nq ,Nq)];
            snk_spin_weights_r2_1[index_3d_cpp(1,nB1+Nw*nB2,nq ,Nw2,Nq)] = src_spin_weights_r2[index_2d_cpp(nB2,nq ,Nq)];
            snk_color_weights_r2_2[index_3d_cpp(0,nB1+Nw*nB2,nq ,Nw2,Nq)] = src_color_weights_r2[index_2d_cpp(nB1,nq ,Nq)];
            snk_spin_weights_r2_2[index_3d_cpp(0,nB1+Nw*nB2,nq ,Nw2,Nq)] = src_spin_weights_r2[index_2d_cpp(nB1,nq ,Nq)];
            snk_color_weights_r2_2[index_3d_cpp(1,nB1+Nw*nB2,nq ,Nw2,Nq)] = src_color_weights_r1[index_2d_cpp(nB2,nq ,Nq)];
            snk_spin_weights_r2_2[index_3d_cpp(1,nB1+Nw*nB2,nq ,Nw2,Nq)] = src_spin_weights_r1[index_2d_cpp(nB2,nq ,Nq)];
            // T1g_r3 
            snk_color_weights_r3[index_3d_cpp(0,nB1+Nw*nB2,nq ,Nw2,Nq)] = src_color_weights_r2[index_2d_cpp(nB1,nq ,Nq)];
            snk_spin_weights_r3[index_3d_cpp(0,nB1+Nw*nB2,nq ,Nw2,Nq)] = src_spin_weights_r2[index_2d_cpp(nB1,nq ,Nq)];
            snk_color_weights_r3[index_3d_cpp(1,nB1+Nw*nB2,nq ,Nw2,Nq)] = src_color_weights_r2[index_2d_cpp(nB2,nq ,Nq)];
            snk_spin_weights_r3[index_3d_cpp(1,nB1+Nw*nB2,nq ,Nw2,Nq)] = src_spin_weights_r2[index_2d_cpp(nB2,nq ,Nq)];
         }
      }
   }
   int snk_1_nq[Nb];
   int snk_2_nq[Nb];
   int snk_3_nq[Nb];
   int snk_1_b[Nb];
   int snk_2_b[Nb];
   int snk_3_b[Nb];
   int snk_1[Nb];
   int snk_2[Nb];
   int snk_3[Nb];
   for (int nperm=0; nperm<Nperms; nperm++) {
      for (int b=0; b<Nb; b++) {
         snk_1[b] = perms[index_2d_cpp(nperm,Nq*b+0 ,2*Nq)] - 1;
         snk_2[b] = perms[index_2d_cpp(nperm,Nq*b+1 ,2*Nq)] - 1;
         snk_3[b] = perms[index_2d_cpp(nperm,Nq*b+2 ,2*Nq)] - 1;
         snk_1_b[b] = (snk_1[b] - snk_1[b] % Nq) / Nq;
         snk_2_b[b] = (snk_2[b] - snk_2[b] % Nq) / Nq;
         snk_3_b[b] = (snk_3[b] - snk_3[b] % Nq) / Nq;
         snk_1_nq[b] = snk_1[b] % Nq;
         snk_2_nq[b] = snk_2[b] % Nq;
         snk_3_nq[b] = snk_3[b] % Nq;
         b_snk_b(b, 0, nperm) = snk_1_b[b];
         b_snk_b(b, 1, nperm) = snk_2_b[b];
         b_snk_b(b, 2, nperm) = snk_3_b[b];
      }
      for (int wnum=0; wnum< Nw2; wnum++) {
         b_snk_color_weights(0, 0, wnum, nperm, 0) = snk_color_weights_r1[index_3d_cpp(snk_1_b[0],wnum,snk_1_nq[0] ,Nw2,Nq)];
         b_snk_spin_weights(0, 0, wnum, nperm, 0) = snk_spin_weights_r1[index_3d_cpp(snk_1_b[0],wnum,snk_1_nq[0] ,Nw2,Nq)];
         b_snk_color_weights(0, 1, wnum, nperm, 0) = snk_color_weights_r1[index_3d_cpp(snk_2_b[0],wnum,snk_2_nq[0] ,Nw2,Nq)];
         b_snk_spin_weights(0, 1, wnum, nperm, 0) = snk_spin_weights_r1[index_3d_cpp(snk_2_b[0],wnum,snk_2_nq[0] ,Nw2,Nq)];
         b_snk_color_weights(0, 2, wnum, nperm, 0) = snk_color_weights_r1[index_3d_cpp(snk_3_b[0],wnum,snk_3_nq[0] ,Nw2,Nq)];
         b_snk_spin_weights(0, 2, wnum, nperm, 0) = snk_spin_weights_r1[index_3d_cpp(snk_3_b[0],wnum,snk_3_nq[0] ,Nw2,Nq)];
         b_snk_color_weights(1, 0, wnum, nperm, 0) = snk_color_weights_r1[index_3d_cpp(snk_1_b[1],wnum,snk_1_nq[1] ,Nw2,Nq)];
         b_snk_spin_weights(1, 0, wnum, nperm, 0) = snk_spin_weights_r1[index_3d_cpp(snk_1_b[1],wnum,snk_1_nq[1] ,Nw2,Nq)];
         b_snk_color_weights(1, 1, wnum, nperm, 0) = snk_color_weights_r1[index_3d_cpp(snk_2_b[1],wnum,snk_2_nq[1] ,Nw2,Nq)];
         b_snk_spin_weights(1, 1, wnum, nperm, 0) = snk_spin_weights_r1[index_3d_cpp(snk_2_b[1],wnum,snk_2_nq[1] ,Nw2,Nq)];
         b_snk_color_weights(1, 2, wnum, nperm, 0) = snk_color_weights_r1[index_3d_cpp(snk_3_b[1],wnum,snk_3_nq[1] ,Nw2,Nq)];
         b_snk_spin_weights(1, 2, wnum, nperm, 0) = snk_spin_weights_r1[index_3d_cpp(snk_3_b[1],wnum,snk_3_nq[1] ,Nw2,Nq)]; 

         b_snk_color_weights(0, 0, wnum, nperm, 1) = snk_color_weights_r3[index_3d_cpp(snk_1_b[0],wnum,snk_1_nq[0] ,Nw2,Nq)];
         b_snk_spin_weights(0, 0, wnum, nperm, 1) = snk_spin_weights_r3[index_3d_cpp(snk_1_b[0],wnum,snk_1_nq[0] ,Nw2,Nq)];
         b_snk_color_weights(0, 1, wnum, nperm, 1) = snk_color_weights_r3[index_3d_cpp(snk_2_b[0],wnum,snk_2_nq[0] ,Nw2,Nq)];
         b_snk_spin_weights(0, 1, wnum, nperm, 1) = snk_spin_weights_r3[index_3d_cpp(snk_2_b[0],wnum,snk_2_nq[0] ,Nw2,Nq)];
         b_snk_color_weights(0, 2, wnum, nperm, 1) = snk_color_weights_r3[index_3d_cpp(snk_3_b[0],wnum,snk_3_nq[0] ,Nw2,Nq)];
         b_snk_spin_weights(0, 2, wnum, nperm, 1) = snk_spin_weights_r3[index_3d_cpp(snk_3_b[0],wnum,snk_3_nq[0] ,Nw2,Nq)];
         b_snk_color_weights(1, 0, wnum, nperm, 1) = snk_color_weights_r3[index_3d_cpp(snk_1_b[1],wnum,snk_1_nq[1] ,Nw2,Nq)];
         b_snk_spin_weights(1, 0, wnum, nperm, 1) = snk_spin_weights_r3[index_3d_cpp(snk_1_b[1],wnum,snk_1_nq[1] ,Nw2,Nq)];
         b_snk_color_weights(1, 1, wnum, nperm, 1) = snk_color_weights_r3[index_3d_cpp(snk_2_b[1],wnum,snk_2_nq[1] ,Nw2,Nq)];
         b_snk_spin_weights(1, 1, wnum, nperm, 1) = snk_spin_weights_r3[index_3d_cpp(snk_2_b[1],wnum,snk_2_nq[1] ,Nw2,Nq)];
         b_snk_color_weights(1, 2, wnum, nperm, 1) = snk_color_weights_r3[index_3d_cpp(snk_3_b[1],wnum,snk_3_nq[1] ,Nw2,Nq)];
         b_snk_spin_weights(1, 2, wnum, nperm, 1) = snk_spin_weights_r3[index_3d_cpp(snk_3_b[1],wnum,snk_3_nq[1] ,Nw2,Nq)]; 

         b_snk_color_weights(0, 0, wnum, nperm, 2) = snk_color_weights_r2_1[index_3d_cpp(snk_1_b[0],wnum,snk_1_nq[0] ,Nw2,Nq)];
         b_snk_spin_weights(0, 0, wnum, nperm, 2) = snk_spin_weights_r2_1[index_3d_cpp(snk_1_b[0],wnum,snk_1_nq[0] ,Nw2,Nq)];
         b_snk_color_weights(0, 1, wnum, nperm, 2) = snk_color_weights_r2_1[index_3d_cpp(snk_2_b[0],wnum,snk_2_nq[0] ,Nw2,Nq)];
         b_snk_spin_weights(0, 1, wnum, nperm, 2) = snk_spin_weights_r2_1[index_3d_cpp(snk_2_b[0],wnum,snk_2_nq[0] ,Nw2,Nq)];
         b_snk_color_weights(0, 2, wnum, nperm, 2) = snk_color_weights_r2_1[index_3d_cpp(snk_3_b[0],wnum,snk_3_nq[0] ,Nw2,Nq)];
         b_snk_spin_weights(0, 2, wnum, nperm, 2) = snk_spin_weights_r2_1[index_3d_cpp(snk_3_b[0],wnum,snk_3_nq[0] ,Nw2,Nq)];
         b_snk_color_weights(1, 0, wnum, nperm, 2) = snk_color_weights_r2_1[index_3d_cpp(snk_1_b[1],wnum,snk_1_nq[1] ,Nw2,Nq)];
         b_snk_spin_weights(1, 0, wnum, nperm, 2) = snk_spin_weights_r2_1[index_3d_cpp(snk_1_b[1],wnum,snk_1_nq[1] ,Nw2,Nq)];
         b_snk_color_weights(1, 1, wnum, nperm, 2) = snk_color_weights_r2_1[index_3d_cpp(snk_2_b[1],wnum,snk_2_nq[1] ,Nw2,Nq)];
         b_snk_spin_weights(1, 1, wnum, nperm, 2) = snk_spin_weights_r2_1[index_3d_cpp(snk_2_b[1],wnum,snk_2_nq[1] ,Nw2,Nq)];
         b_snk_color_weights(1, 2, wnum, nperm, 2) = snk_color_weights_r2_1[index_3d_cpp(snk_3_b[1],wnum,snk_3_nq[1] ,Nw2,Nq)];
         b_snk_spin_weights(1, 2, wnum, nperm, 2) = snk_spin_weights_r2_1[index_3d_cpp(snk_3_b[1],wnum,snk_3_nq[1] ,Nw2,Nq)];

         b_snk_color_weights(0, 0, wnum, nperm, 3) = snk_color_weights_r2_1[index_3d_cpp(snk_1_b[0],wnum,snk_1_nq[0] ,Nw2,Nq)];
         b_snk_spin_weights(0, 0, wnum, nperm, 3) = snk_spin_weights_r2_1[index_3d_cpp(snk_1_b[0],wnum,snk_1_nq[0] ,Nw2,Nq)];
         b_snk_color_weights(0, 1, wnum, nperm, 3) = snk_color_weights_r2_1[index_3d_cpp(snk_2_b[0],wnum,snk_2_nq[0] ,Nw2,Nq)];
         b_snk_spin_weights(0, 1, wnum, nperm, 3) = snk_spin_weights_r2_1[index_3d_cpp(snk_2_b[0],wnum,snk_2_nq[0] ,Nw2,Nq)];
         b_snk_color_weights(0, 2, wnum, nperm, 3) = snk_color_weights_r2_1[index_3d_cpp(snk_3_b[0],wnum,snk_3_nq[0] ,Nw2,Nq)];
         b_snk_spin_weights(0, 2, wnum, nperm, 3) = snk_spin_weights_r2_1[index_3d_cpp(snk_3_b[0],wnum,snk_3_nq[0] ,Nw2,Nq)];
         b_snk_color_weights(1, 0, wnum, nperm, 3) = snk_color_weights_r2_1[index_3d_cpp(snk_1_b[1],wnum,snk_1_nq[1] ,Nw2,Nq)];
         b_snk_spin_weights(1, 0, wnum, nperm, 3) = snk_spin_weights_r2_1[index_3d_cpp(snk_1_b[1],wnum,snk_1_nq[1] ,Nw2,Nq)];
         b_snk_color_weights(1, 1, wnum, nperm, 3) = snk_color_weights_r2_1[index_3d_cpp(snk_2_b[1],wnum,snk_2_nq[1] ,Nw2,Nq)];
         b_snk_spin_weights(1, 1, wnum, nperm, 3) = snk_spin_weights_r2_1[index_3d_cpp(snk_2_b[1],wnum,snk_2_nq[1] ,Nw2,Nq)];
         b_snk_color_weights(1, 2, wnum, nperm, 3) = snk_color_weights_r2_1[index_3d_cpp(snk_3_b[1],wnum,snk_3_nq[1] ,Nw2,Nq)];
         b_snk_spin_weights(1, 2, wnum, nperm, 3) = snk_spin_weights_r2_1[index_3d_cpp(snk_3_b[1],wnum,snk_3_nq[1] ,Nw2,Nq)];

         b_snk_color_weights(0, 0, wnum, nperm, 4) = snk_color_weights_r2_2[index_3d_cpp(snk_1_b[0],wnum,snk_1_nq[0] ,Nw2,Nq)];
         b_snk_spin_weights(0, 0, wnum, nperm, 4) = snk_spin_weights_r2_2[index_3d_cpp(snk_1_b[0],wnum,snk_1_nq[0] ,Nw2,Nq)];
         b_snk_color_weights(0, 1, wnum, nperm, 4) = snk_color_weights_r2_2[index_3d_cpp(snk_2_b[0],wnum,snk_2_nq[0] ,Nw2,Nq)];
         b_snk_spin_weights(0, 1, wnum, nperm, 4) = snk_spin_weights_r2_2[index_3d_cpp(snk_2_b[0],wnum,snk_2_nq[0] ,Nw2,Nq)];
         b_snk_color_weights(0, 2, wnum, nperm, 4) = snk_color_weights_r2_2[index_3d_cpp(snk_3_b[0],wnum,snk_3_nq[0] ,Nw2,Nq)];
         b_snk_spin_weights(0, 2, wnum, nperm, 4) = snk_spin_weights_r2_2[index_3d_cpp(snk_3_b[0],wnum,snk_3_nq[0] ,Nw2,Nq)];
         b_snk_color_weights(1, 0, wnum, nperm, 4) = snk_color_weights_r2_2[index_3d_cpp(snk_1_b[1],wnum,snk_1_nq[1] ,Nw2,Nq)];
         b_snk_spin_weights(1, 0, wnum, nperm, 4) = snk_spin_weights_r2_2[index_3d_cpp(snk_1_b[1],wnum,snk_1_nq[1] ,Nw2,Nq)];
         b_snk_color_weights(1, 1, wnum, nperm, 4) = snk_color_weights_r2_2[index_3d_cpp(snk_2_b[1],wnum,snk_2_nq[1] ,Nw2,Nq)];
         b_snk_spin_weights(1, 1, wnum, nperm, 4) = snk_spin_weights_r2_2[index_3d_cpp(snk_2_b[1],wnum,snk_2_nq[1] ,Nw2,Nq)];
         b_snk_color_weights(1, 2, wnum, nperm, 4) = snk_color_weights_r2_2[index_3d_cpp(snk_3_b[1],wnum,snk_3_nq[1] ,Nw2,Nq)];
         b_snk_spin_weights(1, 2, wnum, nperm, 4) = snk_spin_weights_r2_2[index_3d_cpp(snk_3_b[1],wnum,snk_3_nq[1] ,Nw2,Nq)]; 

         b_snk_color_weights(0, 0, wnum, nperm, 5) = snk_color_weights_r2_2[index_3d_cpp(snk_1_b[0],wnum,snk_1_nq[0] ,Nw2,Nq)];
         b_snk_spin_weights(0, 0, wnum, nperm, 5) = snk_spin_weights_r2_2[index_3d_cpp(snk_1_b[0],wnum,snk_1_nq[0] ,Nw2,Nq)];
         b_snk_color_weights(0, 1, wnum, nperm, 5) = snk_color_weights_r2_2[index_3d_cpp(snk_2_b[0],wnum,snk_2_nq[0] ,Nw2,Nq)];
         b_snk_spin_weights(0, 1, wnum, nperm, 5) = snk_spin_weights_r2_2[index_3d_cpp(snk_2_b[0],wnum,snk_2_nq[0] ,Nw2,Nq)];
         b_snk_color_weights(0, 2, wnum, nperm, 5) = snk_color_weights_r2_2[index_3d_cpp(snk_3_b[0],wnum,snk_3_nq[0] ,Nw2,Nq)];
         b_snk_spin_weights(0, 2, wnum, nperm, 5) = snk_spin_weights_r2_2[index_3d_cpp(snk_3_b[0],wnum,snk_3_nq[0] ,Nw2,Nq)];
         b_snk_color_weights(1, 0, wnum, nperm, 5) = snk_color_weights_r2_2[index_3d_cpp(snk_1_b[1],wnum,snk_1_nq[1] ,Nw2,Nq)];
         b_snk_spin_weights(1, 0, wnum, nperm, 5) = snk_spin_weights_r2_2[index_3d_cpp(snk_1_b[1],wnum,snk_1_nq[1] ,Nw2,Nq)];
         b_snk_color_weights(1, 1, wnum, nperm, 5) = snk_color_weights_r2_2[index_3d_cpp(snk_2_b[1],wnum,snk_2_nq[1] ,Nw2,Nq)];
         b_snk_spin_weights(1, 1, wnum, nperm, 5) = snk_spin_weights_r2_2[index_3d_cpp(snk_2_b[1],wnum,snk_2_nq[1] ,Nw2,Nq)];
         b_snk_color_weights(1, 2, wnum, nperm, 5) = snk_color_weights_r2_2[index_3d_cpp(snk_3_b[1],wnum,snk_3_nq[1] ,Nw2,Nq)];
         b_snk_spin_weights(1, 2, wnum, nperm, 5) = snk_spin_weights_r2_2[index_3d_cpp(snk_3_b[1],wnum,snk_3_nq[1] ,Nw2,Nq)]; 
      }
   }

   Halide::Buffer<double> b_C_r(Nsnk, Nr, Nsrc, Nt, "C_r");
   Halide::Buffer<double> b_C_i(Nsnk, Nr, Nsrc, Nt, "C_i");

	tiramisu_make_fused_dibaryon_blocks_correlator(
				    b_C_r.raw_buffer(),
				    b_C_i.raw_buffer(),

				    b_B1_prop_r.raw_buffer(),
				    b_B1_prop_i.raw_buffer(),
				    b_B2_prop_r.raw_buffer(),
				    b_B2_prop_i.raw_buffer(),
                b_B1_src_psi_r.raw_buffer(),
                b_B1_src_psi_i.raw_buffer(),
                b_B2_src_psi_r.raw_buffer(),
                b_B2_src_psi_i.raw_buffer(),

				    b_snk_blocks.raw_buffer(),
				    b_sigs.raw_buffer(),
				    b_snk_b.raw_buffer(),
				    b_snk_color_weights.raw_buffer(),
				    b_snk_spin_weights.raw_buffer(),
				    b_snk_weights.raw_buffer(),
				    b_snk_psi_r.raw_buffer(),
				    b_snk_psi_i.raw_buffer());

   for (int m=0; m<Nsrc; m++)
      for (int n=0; n<Nsnk; n++)
         for (int t=0; t<Nt; t++) {
            C_re[index_4d_cpp(0,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] = 1/sqrt(2) * b_C_r(n,2,m,t) / ((Vsrc*Vsrc)*(Vsnk*Vsnk));
            C_re[index_4d_cpp(0,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] -= 1/sqrt(2) * b_C_r(n,3,m,t) / ((Vsrc*Vsrc)*(Vsnk*Vsnk));
            C_re[index_4d_cpp(0,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] -= 1/sqrt(2) * b_C_r(n,4,m,t) / ((Vsrc*Vsrc)*(Vsnk*Vsnk));
            C_re[index_4d_cpp(0,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] += 1/sqrt(2) * b_C_r(n,5,m,t) / ((Vsrc*Vsrc)*(Vsnk*Vsnk));
            C_re[index_4d_cpp(1,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] = b_C_r(n,0,m,t) / ((Vsrc*Vsrc)*(Vsnk*Vsnk));
            C_re[index_4d_cpp(2,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] = 1/sqrt(2) * b_C_r(n,2,m,t) / ((Vsrc*Vsrc)*(Vsnk*Vsnk));
            C_re[index_4d_cpp(2,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] += 1/sqrt(2) * b_C_r(n,3,m,t) / ((Vsrc*Vsrc)*(Vsnk*Vsnk));
            C_re[index_4d_cpp(2,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] += 1/sqrt(2) * b_C_r(n,4,m,t) / ((Vsrc*Vsrc)*(Vsnk*Vsnk));
            C_re[index_4d_cpp(2,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] += 1/sqrt(2) * b_C_r(n,5,m,t) / ((Vsrc*Vsrc)*(Vsnk*Vsnk));
            C_re[index_4d_cpp(3,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] = b_C_r(n,1,m,t) / ((Vsrc*Vsrc)*(Vsnk*Vsnk));
            C_im[index_4d_cpp(0,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] = 1/sqrt(2) * b_C_i(n,2,m,t) / ((Vsrc*Vsrc)*(Vsnk*Vsnk));
            C_im[index_4d_cpp(0,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] -= 1/sqrt(2) * b_C_i(n,3,m,t) / ((Vsrc*Vsrc)*(Vsnk*Vsnk));
            C_im[index_4d_cpp(0,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] -= 1/sqrt(2) * b_C_i(n,4,m,t) / ((Vsrc*Vsrc)*(Vsnk*Vsnk));
            C_im[index_4d_cpp(0,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] += 1/sqrt(2) * b_C_i(n,5,m,t) / ((Vsrc*Vsrc)*(Vsnk*Vsnk));
            C_im[index_4d_cpp(1,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] = b_C_i(n,0,m,t) / ((Vsrc*Vsrc)*(Vsnk*Vsnk));
            C_im[index_4d_cpp(2,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] = 1/sqrt(2) * b_C_i(n,2,m,t) / ((Vsrc*Vsrc)*(Vsnk*Vsnk));
            C_im[index_4d_cpp(2,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] += 1/sqrt(2) * b_C_i(n,3,m,t) / ((Vsrc*Vsrc)*(Vsnk*Vsnk));
            C_im[index_4d_cpp(2,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] += 1/sqrt(2) * b_C_i(n,4,m,t) / ((Vsrc*Vsrc)*(Vsnk*Vsnk));
            C_im[index_4d_cpp(2,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] += 1/sqrt(2) * b_C_i(n,5,m,t) / ((Vsrc*Vsrc)*(Vsnk*Vsnk));
            C_im[index_4d_cpp(3,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] = b_C_i(n,1,m,t) / ((Vsrc*Vsrc)*(Vsnk*Vsnk));
         }
          
}

/*
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
    const int Nperms)
{
    int Nb = 2;

    Halide::Buffer<double> C_re(t_C_re, {Nt, Nsnk, Nsrc});
    Halide::Buffer<double> C_im(t_C_im, {Nt, Nsnk, Nsrc});

    // Blocal
    // B1_Blocal_r1_r: tiramisu real part of Blocal.
    // B1_Blocal_r1_i: tiramisu imaginary part of Blocal.
    Halide::Buffer<double> B1_Blocal_re(t_B1_Blocal_re, {Ns, Nc, Nsrc, Vsnk, Ns, Nc, Ns, Nc, Nt});
    Halide::Buffer<double> B1_Blocal_im(t_B1_Blocal_im, {Ns, Nc, Nsrc, Vsnk, Ns, Nc, Ns, Nc, Nt});

    Halide::Buffer<double> B1_Bsingle_re(t_B1_Bsingle_re, {Vsnk, Ns, Nc, Nsrc, Vsnk, Ns, Nc, Ns, Nc, Nt});
    Halide::Buffer<double> B1_Bsingle_im(t_B1_Bsingle_im, {Vsnk, Ns, Nc, Nsrc, Vsnk, Ns, Nc, Ns, Nc, Nt});

    Halide::Buffer<double> B1_Bdouble_re(t_B1_Bdouble_re, {Vsnk, Ns, Nc, Nsrc, Vsnk, Ns, Nc, Ns, Nc, Nt});
    Halide::Buffer<double> B1_Bdouble_im(t_B1_Bdouble_im, {Vsnk, Ns, Nc, Nsrc, Vsnk, Ns, Nc, Ns, Nc, Nt});

    Halide::Buffer<double> B2_Blocal_re(t_B2_Blocal_re, {Ns, Nc, Nsrc, Vsnk, Ns, Nc, Ns, Nc, Nt});
    Halide::Buffer<double> B2_Blocal_im(t_B2_Blocal_im, {Ns, Nc, Nsrc, Vsnk, Ns, Nc, Ns, Nc, Nt});

    Halide::Buffer<double> B2_Bsingle_re(t_B2_Bsingle_re, {Vsnk, Ns, Nc, Nsrc, Vsnk, Ns, Nc, Ns, Nc, Nt});
    Halide::Buffer<double> B2_Bsingle_im(t_B2_Bsingle_im, {Vsnk, Ns, Nc, Nsrc, Vsnk, Ns, Nc, Ns, Nc, Nt});

    Halide::Buffer<double> B2_Bdouble_re(t_B2_Bdouble_re, {Vsnk, Ns, Nc, Nsrc, Vsnk, Ns, Nc, Ns, Nc, Nt});
    Halide::Buffer<double> B2_Bdouble_im(t_B2_Bdouble_im, {Vsnk, Ns, Nc, Nsrc, Vsnk, Ns, Nc, Ns, Nc, Nt});

    Halide::Buffer<int> perms((int *) t_perms, {2*Nq, Nperms});
    Halide::Buffer<int> sigs((int *) t_sigs, Nperms);
    Halide::Buffer<double> overall_weight((double *) t_overall_weight, 1);
    Halide::Buffer<int> snk_color_weights((int *) t_snk_color_weights, {Nq, Nw*Nw, 2});
    Halide::Buffer<int> snk_spin_weights((int *) t_snk_spin_weights, {Nq, Nw*Nw, 2});
    Halide::Buffer<double> snk_weights((double *) t_snk_weights, Nw*Nw);
    Halide::Buffer<double> snk_psi_re((double *) t_snk_psi_re, {Nsnk, Vsnk, Vsnk});
    Halide::Buffer<double> snk_psi_im((double *) t_snk_psi_im, {Nsnk, Vsnk, Vsnk});

    tiramisu_make_dibaryon_correlator(
				    C_re.raw_buffer(),
				    C_im.raw_buffer(),
				    B1_Blocal_re.raw_buffer(),
				    B1_Blocal_im.raw_buffer(),
				    B1_Bsingle_re.raw_buffer(),
				    B1_Bsingle_im.raw_buffer(),
				    B1_Bdouble_re.raw_buffer(),
				    B1_Bdouble_im.raw_buffer(),
				    B2_Blocal_re.raw_buffer(),
				    B2_Blocal_im.raw_buffer(),
				    B2_Bsingle_re.raw_buffer(),
				    B2_Bsingle_im.raw_buffer(),
				    B2_Bdouble_re.raw_buffer(),
				    B2_Bdouble_im.raw_buffer(),
				    perms.raw_buffer(),
				    sigs.raw_buffer(),
				    overall_weight.raw_buffer(),
				    snk_color_weights.raw_buffer(),
				    snk_spin_weights.raw_buffer(),
				    snk_weights.raw_buffer(),
				    snk_psi_re.raw_buffer(),
				    snk_psi_im.raw_buffer());

   FILE *f = fopen("tiramisu", "w");
   fprintf(f, "overall_weight = %lf\n", overall_weight(0));

   for (int t=0; t<Nt; t++)
    for (int iCprime=0; iCprime<Nc; iCprime++)
      for (int iSprime=0; iSprime<Ns; iSprime++)
	for (int kCprime=0; kCprime<Nc; kCprime++)
	  for (int kSprime=0; kSprime<Ns; kSprime++)
	    for (int x=0; x<Vsnk; x++)
	      for (int jCprime=0; jCprime<Nc; jCprime++)
	        for (int jSprime=0; jSprime<Ns; jSprime++)
		  for (int m=0; m<Nsrc; m++)
		     fprintf(f, "B1_Blocal_re = %lf\n", B1_Blocal_re(jSprime, jCprime, m, x, kSprime, kCprime, iSprime, iCprime, t));
   for (int t=0; t<Nt; t++)
    for (int iCprime=0; iCprime<Nc; iCprime++)
      for (int iSprime=0; iSprime<Ns; iSprime++)
	for (int kCprime=0; kCprime<Nc; kCprime++)
	  for (int kSprime=0; kSprime<Ns; kSprime++)
	    for (int x=0; x<Vsnk; x++)
	      for (int jCprime=0; jCprime<Nc; jCprime++)
	        for (int jSprime=0; jSprime<Ns; jSprime++)
		  for (int m=0; m<Nsrc; m++)
		    fprintf(f, "B1_Blocal_im = %lf\n", B1_Blocal_im(jSprime, jCprime, m, x, kSprime, kCprime, iSprime, iCprime, t));
   for (int t=0; t<Nt; t++)
    for (int iCprime=0; iCprime<Nc; iCprime++)
      for (int iSprime=0; iSprime<Ns; iSprime++)
	for (int kCprime=0; kCprime<Nc; kCprime++)
	  for (int kSprime=0; kSprime<Ns; kSprime++)
	    for (int x=0; x<Vsnk; x++)
	      for (int jCprime=0; jCprime<Nc; jCprime++)
	        for (int jSprime=0; jSprime<Ns; jSprime++)
		  for (int m=0; m<Nsrc; m++)
		    fprintf(f, "B2_Blocal_re = %lf\n", B1_Blocal_re(jSprime, jCprime, m, x, kSprime, kCprime, iSprime, iCprime, t));
   for (int t=0; t<Nt; t++)
    for (int iCprime=0; iCprime<Nc; iCprime++)
      for (int iSprime=0; iSprime<Ns; iSprime++)
	for (int kCprime=0; kCprime<Nc; kCprime++)
	  for (int kSprime=0; kSprime<Ns; kSprime++)
	    for (int x=0; x<Vsnk; x++)
	      for (int jCprime=0; jCprime<Nc; jCprime++)
	        for (int jSprime=0; jSprime<Ns; jSprime++)
		  for (int m=0; m<Nsrc; m++)
		    fprintf(f, "B2_Blocal_im = %lf\n", B1_Blocal_im(jSprime, jCprime, m, x, kSprime, kCprime, iSprime, iCprime, t));

        for (int n=0; n<Nsrc; n++)
           for (int iCprime=0; iCprime<Nc; iCprime++)
             for (int iSprime=0; iSprime<Ns; iSprime++)
                for (int jCprime=0; jCprime<Nc; jCprime++)
                   for (int jSprime=0; jSprime<Ns; jSprime++)
                      for (int kCprime=0; kCprime<Nc; kCprime++)
                         for (int kSprime=0; kSprime<Ns; kSprime++)
                            for (int x=0; x<Vsnk; x++)
                              for (int x2=0; x2<Vsnk; x2++)
                                  for (int t=0; t<Nt; t++)
                                     fprintf(f, "B1_Bsingle_re = %lf\n", B1_Bsingle_re(x2, jSprime, jCprime, n, x, kSprime, kCprime, iSprime, iCprime, t));
        for (int n=0; n<Nsrc; n++)
           for (int iCprime=0; iCprime<Nc; iCprime++)
             for (int iSprime=0; iSprime<Ns; iSprime++)
                for (int jCprime=0; jCprime<Nc; jCprime++)
                   for (int jSprime=0; jSprime<Ns; jSprime++)
                      for (int kCprime=0; kCprime<Nc; kCprime++)
                         for (int kSprime=0; kSprime<Ns; kSprime++)
                            for (int x=0; x<Vsnk; x++)
                              for (int x2=0; x2<Vsnk; x2++)
                                  for (int t=0; t<Nt; t++)
                                     fprintf(f, "B1_Bsingle_im = %lf\n", B1_Bsingle_im(x2, jSprime, jCprime, n, x, kSprime, kCprime, iSprime, iCprime, t));
        for (int n=0; n<Nsrc; n++)
           for (int iCprime=0; iCprime<Nc; iCprime++)
             for (int iSprime=0; iSprime<Ns; iSprime++)
                for (int jCprime=0; jCprime<Nc; jCprime++)
                   for (int jSprime=0; jSprime<Ns; jSprime++)
                      for (int kCprime=0; kCprime<Nc; kCprime++)
                         for (int kSprime=0; kSprime<Ns; kSprime++)
                            for (int x=0; x<Vsnk; x++)
                              for (int x2=0; x2<Vsnk; x2++)
                                  for (int t=0; t<Nt; t++)
				    fprintf(f, "B1_Bdouble_re = %lf\n", B1_Bdouble_re(x2, jSprime, jCprime, n, x, kSprime, kCprime, iSprime, iCprime, t));
        for (int n=0; n<Nsrc; n++)
           for (int iCprime=0; iCprime<Nc; iCprime++)
             for (int iSprime=0; iSprime<Ns; iSprime++)
                for (int jCprime=0; jCprime<Nc; jCprime++)
                   for (int jSprime=0; jSprime<Ns; jSprime++)
                      for (int kCprime=0; kCprime<Nc; kCprime++)
                         for (int kSprime=0; kSprime<Ns; kSprime++)
                            for (int x=0; x<Vsnk; x++)
                              for (int x2=0; x2<Vsnk; x2++)
                                  for (int t=0; t<Nt; t++)
                                     fprintf(f, "B1_Bdouble_im = %lf\n", B1_Bdouble_im(x2, jSprime, jCprime, n, x, kSprime, kCprime, iSprime, iCprime, t));
        for (int n=0; n<Nsrc; n++)
           for (int iCprime=0; iCprime<Nc; iCprime++)
             for (int iSprime=0; iSprime<Ns; iSprime++)
                for (int jCprime=0; jCprime<Nc; jCprime++)
                   for (int jSprime=0; jSprime<Ns; jSprime++)
                      for (int kCprime=0; kCprime<Nc; kCprime++)
                         for (int kSprime=0; kSprime<Ns; kSprime++)
                            for (int x=0; x<Vsnk; x++)
                              for (int x2=0; x2<Vsnk; x2++)
                                  for (int t=0; t<Nt; t++)
                                     fprintf(f, "B2_Bsingle_re = %lf\n", B2_Bsingle_re(x2, jSprime, jCprime, n, x, kSprime, kCprime, iSprime, iCprime, t));
        for (int n=0; n<Nsrc; n++)
           for (int iCprime=0; iCprime<Nc; iCprime++)
             for (int iSprime=0; iSprime<Ns; iSprime++)
                for (int jCprime=0; jCprime<Nc; jCprime++)
                   for (int jSprime=0; jSprime<Ns; jSprime++)
                      for (int kCprime=0; kCprime<Nc; kCprime++)
                         for (int kSprime=0; kSprime<Ns; kSprime++)
                            for (int x=0; x<Vsnk; x++)
                              for (int x2=0; x2<Vsnk; x2++)
                                  for (int t=0; t<Nt; t++)
                                     fprintf(f, "B2_Bsingle_im = %lf\n", B2_Bsingle_im(x2, jSprime, jCprime, n, x, kSprime, kCprime, iSprime, iCprime, t));
        for (int n=0; n<Nsrc; n++)
           for (int iCprime=0; iCprime<Nc; iCprime++)
             for (int iSprime=0; iSprime<Ns; iSprime++)
                for (int jCprime=0; jCprime<Nc; jCprime++)
                   for (int jSprime=0; jSprime<Ns; jSprime++)
                      for (int kCprime=0; kCprime<Nc; kCprime++)
                         for (int kSprime=0; kSprime<Ns; kSprime++)
                            for (int x=0; x<Vsnk; x++)
                              for (int x2=0; x2<Vsnk; x2++)
                                  for (int t=0; t<Nt; t++)
				    fprintf(f, "B2_Bdouble_re = %lf\n", B2_Bdouble_re(x2, jSprime, jCprime, n, x, kSprime, kCprime, iSprime, iCprime, t));
        for (int n=0; n<Nsrc; n++)
           for (int iCprime=0; iCprime<Nc; iCprime++)
             for (int iSprime=0; iSprime<Ns; iSprime++)
                for (int jCprime=0; jCprime<Nc; jCprime++)
                   for (int jSprime=0; jSprime<Ns; jSprime++)
                      for (int kCprime=0; kCprime<Nc; kCprime++)
                         for (int kSprime=0; kSprime<Ns; kSprime++)
                            for (int x=0; x<Vsnk; x++)
                              for (int x2=0; x2<Vsnk; x2++)
                                  for (int t=0; t<Nt; t++)
                                     fprintf(f, "B2_Bdouble_im = %lf\n", B2_Bdouble_im(x2, jSprime, jCprime, n, x, kSprime, kCprime, iSprime, iCprime, t));


   for (int nperm=0; nperm<Nperms; nperm++)
	 fprintf(f, "sigs[nperm] = %d\n", sigs(nperm));

   for (int wnum=0; wnum< Nw*Nw; wnum++)
	 fprintf(f, "snk_weights[wnum] = %lf\n", snk_weights(wnum));

   for (int x1 = 0; x1<Vsnk; x1++)
     for (int x2 = 0; x2<Vsnk; x2++)
       for (int n = 0; n<Nsnk; n++)
         fprintf(f, "snk_psi_re[x1,x2,n] = %lf\n", snk_psi_re(n, x2, x1));

   for (int x1 = 0; x1<Vsnk; x1++)
     for (int x2 = 0; x2<Vsnk; x2++)
       for (int n = 0; n<Nsnk; n++)
         fprintf(f, "snk_psi_im[x1,x2,n] = %lf\n", snk_psi_im(n, x2, x1));


   for (int t=0; t<Nt; t++)
     for (int m=0; m<Nsrc; m++)
       for (int n=0; n<Nsnk; n++)
         fprintf(f, "C_re = %lf\n", C_re(t, n, m));

   for (int t=0; t<Nt; t++)
     for (int m=0; m<Nsrc; m++)
       for (int n=0; n<Nsnk; n++)
         fprintf(f, "C_im = %lf\n", C_im(t, n, m));

    fclose(f);
}
*/

#ifdef __cplusplus
}  // extern "C"
#endif

//#endif
