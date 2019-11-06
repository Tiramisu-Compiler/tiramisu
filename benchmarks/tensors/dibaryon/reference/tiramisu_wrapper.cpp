#include "qblocks_2pt_scalar.h"
#include <stdio.h>

#if USE_REFERENCE

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
#else

////////////////////////////////////////////////////////////
#include <cstdlib>
#include <iostream>
#include <complex>

#include "Halide.h"
#include <tiramisu/utils.h>

#include "tiramisu_generated_wrapper.h"

#ifdef __cplusplus
extern "C" {
#endif

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
}

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

/*   for (int b=0; b<Nb; b++) {
         fprintf(f, "snk_1[b] = %d\n", snk_1(b));
         fprintf(f, "snk_1_b[b] = %d\n", snk_1_b(b));
         fprintf(f, "snk_1_nq[b] = %d\n", snk_1_nq(b));
   }*/

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

/*   fprintf(f, "term_re = %lf\n", term_re(0));
   fprintf(f, "term_im = %lf\n", term_im(0)); */

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

#ifdef __cplusplus
}  // extern "C"
#endif

#endif
