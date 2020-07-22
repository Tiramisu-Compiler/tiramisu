#include "Halide.h"
#include <tiramisu/utils.h>
#include <tiramisu/mpi_comm.h>
#include <cstdlib>
#include <iostream>
#include <complex>
#include "benchmarks.h"

#ifdef __cplusplus
extern "C" {
#endif

#include "tiramisu_generated_wrapper.h"

/* index functions */
int index_2d(int a, int b, int length2) {
   return b +length2*( a );
}
int index_3d(int a, int b, int c, int length2, int length3) {
   return c +length3*( b +length2*( a ));
}
int index_4d(int a, int b, int c, int d, int length2, int length3, int length4) {
   return d +length4*( c +length3*( b +length2*( a )));
}
int prop_index(int q, int t, int c1, int s1, int c2, int s2, int y, int x, int Nc, int Ns, int Vsrc, int Vsnk, int Nt) {
   return y +Vsrc*( x +Vsnk*( s1 +Ns*( c1 +Nc*( s2 +Ns*( c2 +Nc*( t +Nt* q ))))));
}

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
    const int NsnkHex,
    const int Nr)
{

   int q, t, iC, iS, jC, jS, y, x, m, n, k, wnum;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
    long mega = 1024*1024;
    std::cout << "Array sizes" << std::endl;
    std::cout << "Prop:" <<  std::endl;
    std::cout << "	Max index size = " << Nq*Vsnk*Vsrc*Nc*Ns*Nc*Ns*Lt <<  std::endl;
    std::cout << "	Array size = " << Nq*Vsnk*Vsrc*Nc*Ns*Nc*Ns*Lt*sizeof(std::complex<double>)/mega << " Mega bytes" << std::endl;
    std::cout << "Q:" <<  std::endl;
    std::cout << "	Max index size = " << Vsnk*Vsrc*Nc*Ns*Nc*Ns <<  std::endl;
    std::cout << "	Array size = " << Vsnk*Vsrc*Nc*Ns*Nc*Ns*sizeof(std::complex<double>)/mega << " Mega bytes" <<  std::endl;

    long kilo = 1024;
    std::cout << "Blocal:" <<  std::endl;
    std::cout << "	Max index size = " << Vsnk*NsrcHex*Nc*Ns*Nc*Ns*Nc*Ns <<  std::endl;
    std::cout << "	Array size = " << Vsnk*NsrcHex*Nc*Ns*Nc*Ns*Nc*Ns*sizeof(std::complex<double>)/kilo << " kilo bytes" <<  std::endl;
    std::cout << std::endl;
    }

   // Halide buffers
   Halide::Buffer<double> b_C_r(NsnkHex, Nr, NsrcHex, Vsnk, Lt, "C_r");
   Halide::Buffer<double> b_C_i(NsnkHex, Nr, NsrcHex, Vsnk, Lt, "C_i");

   Halide::Buffer<int> b_src_color_weights(Nq, Nw, Nr, "src_color_weights");
   Halide::Buffer<int> b_src_spin_weights(Nq, Nw, Nr, "src_spin_weights");
   Halide::Buffer<double> b_src_weights(Nw, Nr, "src_weights");

   Halide::Buffer<int> b_snk_blocks(Nr, "snk_blocks");
   Halide::Buffer<int> b_snk_color_weights(Nq, Nw, Nr, "snk_color_weights");
   Halide::Buffer<int> b_snk_spin_weights(Nq, Nw, Nr, "snk_spin_weights");
   Halide::Buffer<double> b_snk_weights(Nw, Nr, "snk_weights");

    // prop
    Halide::Buffer<double> b_B1_prop_r((double *)B1_prop_re, {Vsrc, Vsnk, Ns, Nc, Ns, Nc, Lt, Nq});
    Halide::Buffer<double> b_B1_prop_i((double *)B1_prop_im, {Vsrc, Vsnk, Ns, Nc, Ns, Nc, Lt, Nq});

    if (rank == 0)
   printf("prop elem %4.9f \n", b_B1_prop_r(0,0,0,0,0,0,0,0));

    // psi
    Halide::Buffer<double> b_B1_src_psi_r((double *)src_psi_B1_re, {NsrcHex, Vsrc});
    Halide::Buffer<double> b_B1_src_psi_i((double *)src_psi_B1_im, {NsrcHex, Vsrc});
    Halide::Buffer<double> b_B1_snk_psi_r((double *)snk_psi_B1_re, {NsnkHex, Vsnk});
    Halide::Buffer<double> b_B1_snk_psi_i((double *)snk_psi_B1_im, {NsnkHex, Vsnk});

   // Weights
   int* snk_color_weights_r1 = (int *) malloc(Nw * Nq * sizeof (int));
   int* snk_color_weights_r2 = (int *) malloc(Nw * Nq * sizeof (int));
   int* snk_spin_weights_r1 = (int *) malloc(Nw * Nq * sizeof (int));
   int* snk_spin_weights_r2 = (int *) malloc(Nw * Nq * sizeof (int));
   for (int nB1=0; nB1<Nw; nB1++) {
         b_src_weights(nB1, 0) = src_weights_r1[nB1];
         b_src_weights(nB1, 1) = src_weights_r2[nB1];
         b_snk_weights(nB1, 0) = src_weights_r1[nB1];
         b_snk_weights(nB1, 1) = src_weights_r2[nB1];
         for (int nq=0; nq<Nq; nq++) {
            // G1g_r1
            snk_color_weights_r1[index_2d(nB1,nq ,Nq)] = src_color_weights_r1[index_2d(nB1,nq ,Nq)];
            snk_spin_weights_r1[index_2d(nB1,nq ,Nq)] = src_spin_weights_r1[index_2d(nB1,nq ,Nq)];
            // G1g_r2 
            snk_color_weights_r2[index_2d(nB1,nq ,Nq)] = src_color_weights_r2[index_2d(nB1,nq ,Nq)];
            snk_spin_weights_r2[index_2d(nB1,nq ,Nq)] = src_spin_weights_r2[index_2d(nB1,nq ,Nq)];
         }
   }
   b_snk_blocks(0) = 1;
   b_snk_blocks(1) = 2;
      for (int wnum=0; wnum< Nw; wnum++) {
         b_src_color_weights(0, wnum, 0) = snk_color_weights_r1[index_2d(wnum,0 ,Nq)];
         b_src_spin_weights(0, wnum, 0) = snk_spin_weights_r1[index_2d(wnum,0 ,Nq)];
         b_src_color_weights(1, wnum, 0) = snk_color_weights_r1[index_2d(wnum,1 ,Nq)];
         b_src_spin_weights(1, wnum, 0) = snk_spin_weights_r1[index_2d(wnum,1 ,Nq)];
         b_src_color_weights(2, wnum, 0) = snk_color_weights_r1[index_2d(wnum,2 ,Nq)];
         b_src_spin_weights(2, wnum, 0) = snk_spin_weights_r1[index_2d(wnum,2 ,Nq)];

         b_src_color_weights(0, wnum, 1) = snk_color_weights_r2[index_2d(wnum,0 ,Nq)];
         b_src_spin_weights(0, wnum, 1) = snk_spin_weights_r2[index_2d(wnum,0 ,Nq)];
         b_src_color_weights(1, wnum, 1) = snk_color_weights_r2[index_2d(wnum,1 ,Nq)];
         b_src_spin_weights(1, wnum, 1) = snk_spin_weights_r2[index_2d(wnum,1 ,Nq)];
         b_src_color_weights(2, wnum, 1) = snk_color_weights_r2[index_2d(wnum,2 ,Nq)];
         b_src_spin_weights(2, wnum, 1) = snk_spin_weights_r2[index_2d(wnum,2 ,Nq)];

         b_snk_color_weights(0, wnum, 0) = snk_color_weights_r1[index_2d(wnum,0 ,Nq)];
         b_snk_spin_weights(0, wnum, 0) = snk_spin_weights_r1[index_2d(wnum,0 ,Nq)];
         b_snk_color_weights(1, wnum, 0) = snk_color_weights_r1[index_2d(wnum,1 ,Nq)];
         b_snk_spin_weights(1, wnum, 0) = snk_spin_weights_r1[index_2d(wnum,1 ,Nq)];
         b_snk_color_weights(2, wnum, 0) = snk_color_weights_r1[index_2d(wnum,2 ,Nq)];
         b_snk_spin_weights(2, wnum, 0) = snk_spin_weights_r1[index_2d(wnum,2 ,Nq)];

         b_snk_color_weights(0, wnum, 1) = snk_color_weights_r2[index_2d(wnum,0 ,Nq)];
         b_snk_spin_weights(0, wnum, 1) = snk_spin_weights_r2[index_2d(wnum,0 ,Nq)];
         b_snk_color_weights(1, wnum, 1) = snk_color_weights_r2[index_2d(wnum,1 ,Nq)];
         b_snk_spin_weights(1, wnum, 1) = snk_spin_weights_r2[index_2d(wnum,1 ,Nq)];
         b_snk_color_weights(2, wnum, 1) = snk_color_weights_r2[index_2d(wnum,2 ,Nq)];
         b_snk_spin_weights(2, wnum, 1) = snk_spin_weights_r2[index_2d(wnum,2 ,Nq)];
      }

   for (int b=0; b<Nr; b++)
      for (int m=0; m<NsrcHex; m++)
         for (int n=0; n<NsnkHex; n++)
               for (int x=0; x<Vsnk; x++) {
                  b_C_r(n,b,m,x,t) = 0.0;
                  b_C_i(n,b,m,x,t) = 0.0;
            } 

   if (rank == 0) {
   printf("prop 1 %4.16f + I %4.16f \n", b_B1_prop_r(0,0,0,0,0,0,0,0), b_B1_prop_i(0,0,0,0,0,0,0,0));
   printf("psi src 1 %4.16f + I %4.16f \n", b_B1_src_psi_r(0,0), b_B1_src_psi_i(0,0));
   printf("psi snk %4.16f + I %4.16f \n", b_B1_snk_psi_r(0,0,0), b_B1_snk_psi_i(0,0,0));
   printf("weights snk %4.16f \n", b_snk_weights(0,0));
   }
   tiramisu_make_fused_baryon_blocks_correlator(
				    b_C_r.raw_buffer(),
				    b_C_i.raw_buffer(),
				    b_B1_prop_r.raw_buffer(),
				    b_B1_prop_i.raw_buffer(),
                b_B1_src_psi_r.raw_buffer(),
                b_B1_src_psi_i.raw_buffer(),
                b_B1_snk_psi_r.raw_buffer(),
                b_B1_snk_psi_i.raw_buffer(),
				    b_src_color_weights.raw_buffer(),
				    b_src_spin_weights.raw_buffer(),
				    b_src_weights.raw_buffer(),
				    b_snk_blocks.raw_buffer(),
				    b_snk_color_weights.raw_buffer(),
				    b_snk_spin_weights.raw_buffer(),
				    b_snk_weights.raw_buffer());

    // symmetrize and such
      for (int m=0; m<NsrcHex; m++)
         for (int n=0; n<NsnkHex; n++)
            for (int t=0; t<Lt; t++)  {
               double number0r;
               double number0i;
               double number1r;
               double number1i; 
               double this_number0r = b_C_r(n,0,m,rank,t);
               double this_number0i = b_C_i(n,0,m,rank,t);
               double this_number1r = b_C_r(n,1,m,rank,t);
               double this_number1i = b_C_i(n,1,m,rank,t); 
               MPI_Allreduce(&this_number0r, &number0r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
               MPI_Allreduce(&this_number0i, &number0i, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
               MPI_Allreduce(&this_number1r, &number1r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
               MPI_Allreduce(&this_number1i, &number1i, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
               MPI_Barrier(MPI_COMM_WORLD);
                  C_re[index_4d(0,m,n,t, NsrcHex,NsnkHex,Lt)] = number0r;
                  C_re[index_4d(1,m,n,t, NsrcHex,NsnkHex,Lt)] = number1r;
                  C_im[index_4d(0,m,n,t, NsrcHex,NsnkHex,Lt)] = number0i;
                  C_im[index_4d(1,m,n,t, NsrcHex,NsnkHex,Lt)] = number1i; 
               }

   if (rank == 0) {
   for (int b=0; b<2; b++) {
      printf("\n");
      for (int m=0; m<NsrcHex; m++)
         for (int n=0; n<NsnkHex; n++)
            for (int t=0; t<Lt; t++) {
                  printf("r=%d, m=%d, n=%d, t=%d: %4.9e + I (%4.9e) \n", b, m, n, t, C_re[index_4d(b,m,n,t, NsrcHex,NsnkHex,Lt)],  C_im[index_4d(b,m,n,t, NsrcHex,NsnkHex,Lt)]);
            }
   }
   }

   free(snk_color_weights_r1);
   free(snk_color_weights_r2);
   free(snk_spin_weights_r1);
   free(snk_spin_weights_r2);
}

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
    const int *hex_snk_color_weights_A1,
    const int *hex_snk_spin_weights_A1,
    const double *hex_snk_weights_A1,
    const int *hex_snk_color_weights_T1_r1,
    const int *hex_snk_spin_weights_T1_r1,
    const double *hex_snk_weights_T1_r1,
    const int *hex_snk_color_weights_T1_r2,
    const int *hex_snk_spin_weights_T1_r2,
    const double *hex_snk_weights_T1_r2,
    const int *hex_snk_color_weights_T1_r3,
    const int *hex_snk_spin_weights_T1_r3,
    const double *hex_snk_weights_T1_r3,
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
    const int Nperms,
    const int Nr,
    const int Nw2Hex,
    const int Nb,
    const int NEntangled)
{

   int q, t, iC, iS, jC, jS, y, x, x1, x2, m, n, k, wnum, nperm, b;
   int Nw2 = Nw*Nw;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   if (rank == 0) {
    long mega = 1024*1024;
    long kilo = 1024;
    std::cout << "Array sizes" << std::endl;
    std::cout << "Prop:" <<  std::endl;
    std::cout << "	Max index size = " << Nq*Vsnk*Vsrc*Nc*Ns*Nc*Ns*Lt <<  std::endl;
    std::cout << "	Array size = " << Nq*Vsnk*Vsrc*Nc*Ns*Nc*Ns*Lt*sizeof(std::complex<double>)/mega << " Mega bytes" << std::endl;
    std::cout << "Blocal, Bsingle, Bdouble:" <<  std::endl;
    std::cout << "	Max index size = " << Nsrc*Nc*Ns*Nc*Ns*Nc*Ns <<  std::endl;
    std::cout << "	Array size = " << Nsrc*Nc*Ns*Nc*Ns*Nc*Ns*sizeof(std::complex<double>)/kilo << " kilo bytes" <<  std::endl;
    std::cout << std::endl;
  }


   // Halide buffers
   Halide::Buffer<double> b_C_r(Nsnk+NsnkHex, Nr, Nsrc+NsrcHex, Vsnk, Lt, "C_r");
   Halide::Buffer<double> b_C_i(Nsnk+NsnkHex, Nr, Nsrc+NsrcHex, Vsnk, Lt, "C_i");
   //Halide::Buffer<double> b_C_r(Nsnk+NsnkHex, Nr, Nsrc+NsrcHex, Lt, "C_r");
   //Halide::Buffer<double> b_C_i(Nsnk+NsnkHex, Nr, Nsrc+NsrcHex, Lt, "C_i");

   Halide::Buffer<int> b_src_color_weights(Nq, Nw, Nr, "src_color_weights");
   Halide::Buffer<int> b_src_spin_weights(Nq, Nw, Nr, "src_spin_weights");
   Halide::Buffer<double> b_src_weights(Nw, Nr, "src_weights");

   Halide::Buffer<int> b_snk_blocks(2, Nr, "snk_blocks");
   Halide::Buffer<int> b_snk_b(2, Nq, Nperms, "snk_b");
   Halide::Buffer<int> b_snk_color_weights(2, Nq, Nw2, Nperms, Nr, "snk_color_weights");
   Halide::Buffer<int> b_snk_spin_weights(2, Nq, Nw2, Nperms, Nr, "snk_spin_weights");
   Halide::Buffer<double> b_snk_weights(Nw2, Nr, "snk_weights");
   Halide::Buffer<int> b_hex_snk_color_weights(2, Nq, Nw2Hex, Nperms, Nr, "hex_snk_color_weights");
   Halide::Buffer<int> b_hex_snk_spin_weights(2, Nq, Nw2Hex, Nperms, Nr, "hex_snk_spin_weights");
   Halide::Buffer<double> b_hex_snk_weights(Nw2Hex, Nr, "hex_snk_weights");

    // prop
    Halide::Buffer<double> b_B1_prop_r((double *)B1_prop_re, {Vsrc, Vsnk, Ns, Nc, Ns, Nc, Lt, Nq});
    Halide::Buffer<double> b_B1_prop_i((double *)B1_prop_im, {Vsrc, Vsnk, Ns, Nc, Ns, Nc, Lt, Nq});
    Halide::Buffer<double> b_B2_prop_r((double *)B2_prop_re, {Vsrc, Vsnk, Ns, Nc, Ns, Nc, Lt, Nq});
    Halide::Buffer<double> b_B2_prop_i((double *)B2_prop_im, {Vsrc, Vsnk, Ns, Nc, Ns, Nc, Lt, Nq});

    // psi
    Halide::Buffer<double> b_B1_src_psi_r((double *)src_psi_B1_re, {Nsrc, Vsrc});
    Halide::Buffer<double> b_B1_src_psi_i((double *)src_psi_B1_im, {Nsrc, Vsrc});
    Halide::Buffer<double> b_B2_src_psi_r((double *)src_psi_B2_re, {Nsrc, Vsrc});
    Halide::Buffer<double> b_B2_src_psi_i((double *)src_psi_B2_im, {Nsrc, Vsrc});
    Halide::Buffer<double> b_B1_snk_psi_r((double *)snk_psi_B1_re, {Nsnk, Vsnk});
    Halide::Buffer<double> b_B1_snk_psi_i((double *)snk_psi_B1_im, {Nsnk, Vsnk});
    Halide::Buffer<double> b_B2_snk_psi_r((double *)snk_psi_B2_re, {Nsnk, Vsnk});
    Halide::Buffer<double> b_B2_snk_psi_i((double *)snk_psi_B2_im, {Nsnk, Vsnk});
    Halide::Buffer<double> b_hex_src_psi_r((double *)hex_src_psi_re, {NsrcHex, Vsrc});
    Halide::Buffer<double> b_hex_src_psi_i((double *)hex_src_psi_im, {NsrcHex, Vsrc});
    Halide::Buffer<double> b_hex_snk_psi_r((double *)hex_snk_psi_re, {NsnkHex, Vsnk});
    Halide::Buffer<double> b_hex_snk_psi_i((double *)hex_snk_psi_im, {NsnkHex, Vsnk});
    Halide::Buffer<double> b_snk_psi_r((double *)snk_psi_re, {NEntangled, Vsnk, Vsnk});
    Halide::Buffer<double> b_snk_psi_i((double *)snk_psi_im, {NEntangled, Vsnk, Vsnk});

   Halide::Buffer<int> b_sigs((int *)sigs, {Nperms});

   // Weights
      for (int wnum=0; wnum< Nw; wnum++) {
         b_src_weights(wnum, 0) = src_weights_r1[wnum];
         b_src_weights(wnum, 1) = src_weights_r2[wnum];

         b_src_color_weights(0, wnum, 0) = src_color_weights_r1[index_2d(wnum,0 ,Nq)];
         b_src_spin_weights(0, wnum, 0) = src_spin_weights_r1[index_2d(wnum,0 ,Nq)];
         b_src_color_weights(1, wnum, 0) = src_color_weights_r1[index_2d(wnum,1 ,Nq)];
         b_src_spin_weights(1, wnum, 0) = src_spin_weights_r1[index_2d(wnum,1 ,Nq)];
         b_src_color_weights(2, wnum, 0) = src_color_weights_r1[index_2d(wnum,2 ,Nq)];
         b_src_spin_weights(2, wnum, 0) = src_spin_weights_r1[index_2d(wnum,2 ,Nq)];

         b_src_color_weights(0, wnum, 1) = src_color_weights_r2[index_2d(wnum,0 ,Nq)];
         b_src_spin_weights(0, wnum, 1) = src_spin_weights_r2[index_2d(wnum,0 ,Nq)];
         b_src_color_weights(1, wnum, 1) = src_color_weights_r2[index_2d(wnum,1 ,Nq)];
         b_src_spin_weights(1, wnum, 1) = src_spin_weights_r2[index_2d(wnum,1 ,Nq)];
         b_src_color_weights(2, wnum, 1) = src_color_weights_r2[index_2d(wnum,2 ,Nq)];
         b_src_spin_weights(2, wnum, 1) = src_spin_weights_r2[index_2d(wnum,2 ,Nq)];
      }
   int* snk_color_weights_1 = (int *) malloc(2 * Nw2 * Nq * sizeof (int));
   int* snk_color_weights_2 = (int *) malloc(2 * Nw2 * Nq * sizeof (int));
   int* snk_color_weights_r1 = (int *) malloc(2 * Nw2 * Nq * sizeof (int));
   int* snk_color_weights_r2_1 = (int *) malloc(2 * Nw2 * Nq * sizeof (int));
   int* snk_color_weights_r2_2 = (int *) malloc(2 * Nw2 * Nq * sizeof (int));
   int* snk_color_weights_r3 = (int *) malloc(2 * Nw2 * Nq * sizeof (int));
   int* snk_spin_weights_1 = (int *) malloc(2 * Nw2 * Nq * sizeof (int));
   int* snk_spin_weights_2 = (int *) malloc(2 * Nw2 * Nq * sizeof (int));
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
            snk_color_weights_r1[index_3d(0,nB1+Nw*nB2,nq ,Nw2,Nq)] = src_color_weights_r1[index_2d(nB1,nq ,Nq)];
            snk_spin_weights_r1[index_3d(0,nB1+Nw*nB2,nq ,Nw2,Nq)] = src_spin_weights_r1[index_2d(nB1,nq ,Nq)];
            snk_color_weights_r1[index_3d(1,nB1+Nw*nB2,nq ,Nw2,Nq)] = src_color_weights_r1[index_2d(nB2,nq ,Nq)];
            snk_spin_weights_r1[index_3d(1,nB1+Nw*nB2,nq ,Nw2,Nq)] = src_spin_weights_r1[index_2d(nB2,nq ,Nq)];
            // T1g_r2
            snk_color_weights_r2_1[index_3d(0,nB1+Nw*nB2,nq ,Nw2,Nq)] = src_color_weights_r1[index_2d(nB1,nq ,Nq)];
            snk_spin_weights_r2_1[index_3d(0,nB1+Nw*nB2,nq ,Nw2,Nq)] = src_spin_weights_r1[index_2d(nB1,nq ,Nq)];
            snk_color_weights_r2_1[index_3d(1,nB1+Nw*nB2,nq ,Nw2,Nq)] = src_color_weights_r2[index_2d(nB2,nq ,Nq)];
            snk_spin_weights_r2_1[index_3d(1,nB1+Nw*nB2,nq ,Nw2,Nq)] = src_spin_weights_r2[index_2d(nB2,nq ,Nq)];
            snk_color_weights_r2_2[index_3d(0,nB1+Nw*nB2,nq ,Nw2,Nq)] = src_color_weights_r2[index_2d(nB1,nq ,Nq)];
            snk_spin_weights_r2_2[index_3d(0,nB1+Nw*nB2,nq ,Nw2,Nq)] = src_spin_weights_r2[index_2d(nB1,nq ,Nq)];
            snk_color_weights_r2_2[index_3d(1,nB1+Nw*nB2,nq ,Nw2,Nq)] = src_color_weights_r1[index_2d(nB2,nq ,Nq)];
            snk_spin_weights_r2_2[index_3d(1,nB1+Nw*nB2,nq ,Nw2,Nq)] = src_spin_weights_r1[index_2d(nB2,nq ,Nq)];
            // T1g_r3 
            snk_color_weights_r3[index_3d(0,nB1+Nw*nB2,nq ,Nw2,Nq)] = src_color_weights_r2[index_2d(nB1,nq ,Nq)];
            snk_spin_weights_r3[index_3d(0,nB1+Nw*nB2,nq ,Nw2,Nq)] = src_spin_weights_r2[index_2d(nB1,nq ,Nq)];
            snk_color_weights_r3[index_3d(1,nB1+Nw*nB2,nq ,Nw2,Nq)] = src_color_weights_r2[index_2d(nB2,nq ,Nq)];
            snk_spin_weights_r3[index_3d(1,nB1+Nw*nB2,nq ,Nw2,Nq)] = src_spin_weights_r2[index_2d(nB2,nq ,Nq)];
         }
      }
   }
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
         snk_1[b] = perms[index_2d(nperm,Nq*b+0 ,2*Nq)] - 1;
         snk_2[b] = perms[index_2d(nperm,Nq*b+1 ,2*Nq)] - 1;
         snk_3[b] = perms[index_2d(nperm,Nq*b+2 ,2*Nq)] - 1;
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
         b_snk_color_weights(0, 0, wnum, nperm, 0) = snk_color_weights_r1[index_3d(snk_1_b[0],wnum,snk_1_nq[0] ,Nw2,Nq)];
         b_snk_spin_weights(0, 0, wnum, nperm, 0) = snk_spin_weights_r1[index_3d(snk_1_b[0],wnum,snk_1_nq[0] ,Nw2,Nq)];
         b_snk_color_weights(0, 1, wnum, nperm, 0) = snk_color_weights_r1[index_3d(snk_2_b[0],wnum,snk_2_nq[0] ,Nw2,Nq)];
         b_snk_spin_weights(0, 1, wnum, nperm, 0) = snk_spin_weights_r1[index_3d(snk_2_b[0],wnum,snk_2_nq[0] ,Nw2,Nq)];
         b_snk_color_weights(0, 2, wnum, nperm, 0) = snk_color_weights_r1[index_3d(snk_3_b[0],wnum,snk_3_nq[0] ,Nw2,Nq)];
         b_snk_spin_weights(0, 2, wnum, nperm, 0) = snk_spin_weights_r1[index_3d(snk_3_b[0],wnum,snk_3_nq[0] ,Nw2,Nq)];
         b_snk_color_weights(1, 0, wnum, nperm, 0) = snk_color_weights_r1[index_3d(snk_1_b[1],wnum,snk_1_nq[1] ,Nw2,Nq)];
         b_snk_spin_weights(1, 0, wnum, nperm, 0) = snk_spin_weights_r1[index_3d(snk_1_b[1],wnum,snk_1_nq[1] ,Nw2,Nq)];
         b_snk_color_weights(1, 1, wnum, nperm, 0) = snk_color_weights_r1[index_3d(snk_2_b[1],wnum,snk_2_nq[1] ,Nw2,Nq)];
         b_snk_spin_weights(1, 1, wnum, nperm, 0) = snk_spin_weights_r1[index_3d(snk_2_b[1],wnum,snk_2_nq[1] ,Nw2,Nq)];
         b_snk_color_weights(1, 2, wnum, nperm, 0) = snk_color_weights_r1[index_3d(snk_3_b[1],wnum,snk_3_nq[1] ,Nw2,Nq)];
         b_snk_spin_weights(1, 2, wnum, nperm, 0) = snk_spin_weights_r1[index_3d(snk_3_b[1],wnum,snk_3_nq[1] ,Nw2,Nq)]; 

         b_snk_color_weights(0, 0, wnum, nperm, 1) = snk_color_weights_r3[index_3d(snk_1_b[0],wnum,snk_1_nq[0] ,Nw2,Nq)];
         b_snk_spin_weights(0, 0, wnum, nperm, 1) = snk_spin_weights_r3[index_3d(snk_1_b[0],wnum,snk_1_nq[0] ,Nw2,Nq)];
         b_snk_color_weights(0, 1, wnum, nperm, 1) = snk_color_weights_r3[index_3d(snk_2_b[0],wnum,snk_2_nq[0] ,Nw2,Nq)];
         b_snk_spin_weights(0, 1, wnum, nperm, 1) = snk_spin_weights_r3[index_3d(snk_2_b[0],wnum,snk_2_nq[0] ,Nw2,Nq)];
         b_snk_color_weights(0, 2, wnum, nperm, 1) = snk_color_weights_r3[index_3d(snk_3_b[0],wnum,snk_3_nq[0] ,Nw2,Nq)];
         b_snk_spin_weights(0, 2, wnum, nperm, 1) = snk_spin_weights_r3[index_3d(snk_3_b[0],wnum,snk_3_nq[0] ,Nw2,Nq)];
         b_snk_color_weights(1, 0, wnum, nperm, 1) = snk_color_weights_r3[index_3d(snk_1_b[1],wnum,snk_1_nq[1] ,Nw2,Nq)];
         b_snk_spin_weights(1, 0, wnum, nperm, 1) = snk_spin_weights_r3[index_3d(snk_1_b[1],wnum,snk_1_nq[1] ,Nw2,Nq)];
         b_snk_color_weights(1, 1, wnum, nperm, 1) = snk_color_weights_r3[index_3d(snk_2_b[1],wnum,snk_2_nq[1] ,Nw2,Nq)];
         b_snk_spin_weights(1, 1, wnum, nperm, 1) = snk_spin_weights_r3[index_3d(snk_2_b[1],wnum,snk_2_nq[1] ,Nw2,Nq)];
         b_snk_color_weights(1, 2, wnum, nperm, 1) = snk_color_weights_r3[index_3d(snk_3_b[1],wnum,snk_3_nq[1] ,Nw2,Nq)];
         b_snk_spin_weights(1, 2, wnum, nperm, 1) = snk_spin_weights_r3[index_3d(snk_3_b[1],wnum,snk_3_nq[1] ,Nw2,Nq)]; 

         b_snk_color_weights(0, 0, wnum, nperm, 2) = snk_color_weights_r2_1[index_3d(snk_1_b[0],wnum,snk_1_nq[0] ,Nw2,Nq)];
         b_snk_spin_weights(0, 0, wnum, nperm, 2) = snk_spin_weights_r2_1[index_3d(snk_1_b[0],wnum,snk_1_nq[0] ,Nw2,Nq)];
         b_snk_color_weights(0, 1, wnum, nperm, 2) = snk_color_weights_r2_1[index_3d(snk_2_b[0],wnum,snk_2_nq[0] ,Nw2,Nq)];
         b_snk_spin_weights(0, 1, wnum, nperm, 2) = snk_spin_weights_r2_1[index_3d(snk_2_b[0],wnum,snk_2_nq[0] ,Nw2,Nq)];
         b_snk_color_weights(0, 2, wnum, nperm, 2) = snk_color_weights_r2_1[index_3d(snk_3_b[0],wnum,snk_3_nq[0] ,Nw2,Nq)];
         b_snk_spin_weights(0, 2, wnum, nperm, 2) = snk_spin_weights_r2_1[index_3d(snk_3_b[0],wnum,snk_3_nq[0] ,Nw2,Nq)];
         b_snk_color_weights(1, 0, wnum, nperm, 2) = snk_color_weights_r2_1[index_3d(snk_1_b[1],wnum,snk_1_nq[1] ,Nw2,Nq)];
         b_snk_spin_weights(1, 0, wnum, nperm, 2) = snk_spin_weights_r2_1[index_3d(snk_1_b[1],wnum,snk_1_nq[1] ,Nw2,Nq)];
         b_snk_color_weights(1, 1, wnum, nperm, 2) = snk_color_weights_r2_1[index_3d(snk_2_b[1],wnum,snk_2_nq[1] ,Nw2,Nq)];
         b_snk_spin_weights(1, 1, wnum, nperm, 2) = snk_spin_weights_r2_1[index_3d(snk_2_b[1],wnum,snk_2_nq[1] ,Nw2,Nq)];
         b_snk_color_weights(1, 2, wnum, nperm, 2) = snk_color_weights_r2_1[index_3d(snk_3_b[1],wnum,snk_3_nq[1] ,Nw2,Nq)];
         b_snk_spin_weights(1, 2, wnum, nperm, 2) = snk_spin_weights_r2_1[index_3d(snk_3_b[1],wnum,snk_3_nq[1] ,Nw2,Nq)];

         b_snk_color_weights(0, 0, wnum, nperm, 3) = snk_color_weights_r2_1[index_3d(snk_1_b[0],wnum,snk_1_nq[0] ,Nw2,Nq)];
         b_snk_spin_weights(0, 0, wnum, nperm, 3) = snk_spin_weights_r2_1[index_3d(snk_1_b[0],wnum,snk_1_nq[0] ,Nw2,Nq)];
         b_snk_color_weights(0, 1, wnum, nperm, 3) = snk_color_weights_r2_1[index_3d(snk_2_b[0],wnum,snk_2_nq[0] ,Nw2,Nq)];
         b_snk_spin_weights(0, 1, wnum, nperm, 3) = snk_spin_weights_r2_1[index_3d(snk_2_b[0],wnum,snk_2_nq[0] ,Nw2,Nq)];
         b_snk_color_weights(0, 2, wnum, nperm, 3) = snk_color_weights_r2_1[index_3d(snk_3_b[0],wnum,snk_3_nq[0] ,Nw2,Nq)];
         b_snk_spin_weights(0, 2, wnum, nperm, 3) = snk_spin_weights_r2_1[index_3d(snk_3_b[0],wnum,snk_3_nq[0] ,Nw2,Nq)];
         b_snk_color_weights(1, 0, wnum, nperm, 3) = snk_color_weights_r2_1[index_3d(snk_1_b[1],wnum,snk_1_nq[1] ,Nw2,Nq)];
         b_snk_spin_weights(1, 0, wnum, nperm, 3) = snk_spin_weights_r2_1[index_3d(snk_1_b[1],wnum,snk_1_nq[1] ,Nw2,Nq)];
         b_snk_color_weights(1, 1, wnum, nperm, 3) = snk_color_weights_r2_1[index_3d(snk_2_b[1],wnum,snk_2_nq[1] ,Nw2,Nq)];
         b_snk_spin_weights(1, 1, wnum, nperm, 3) = snk_spin_weights_r2_1[index_3d(snk_2_b[1],wnum,snk_2_nq[1] ,Nw2,Nq)];
         b_snk_color_weights(1, 2, wnum, nperm, 3) = snk_color_weights_r2_1[index_3d(snk_3_b[1],wnum,snk_3_nq[1] ,Nw2,Nq)];
         b_snk_spin_weights(1, 2, wnum, nperm, 3) = snk_spin_weights_r2_1[index_3d(snk_3_b[1],wnum,snk_3_nq[1] ,Nw2,Nq)];

         b_snk_color_weights(0, 0, wnum, nperm, 4) = snk_color_weights_r2_2[index_3d(snk_1_b[0],wnum,snk_1_nq[0] ,Nw2,Nq)];
         b_snk_spin_weights(0, 0, wnum, nperm, 4) = snk_spin_weights_r2_2[index_3d(snk_1_b[0],wnum,snk_1_nq[0] ,Nw2,Nq)];
         b_snk_color_weights(0, 1, wnum, nperm, 4) = snk_color_weights_r2_2[index_3d(snk_2_b[0],wnum,snk_2_nq[0] ,Nw2,Nq)];
         b_snk_spin_weights(0, 1, wnum, nperm, 4) = snk_spin_weights_r2_2[index_3d(snk_2_b[0],wnum,snk_2_nq[0] ,Nw2,Nq)];
         b_snk_color_weights(0, 2, wnum, nperm, 4) = snk_color_weights_r2_2[index_3d(snk_3_b[0],wnum,snk_3_nq[0] ,Nw2,Nq)];
         b_snk_spin_weights(0, 2, wnum, nperm, 4) = snk_spin_weights_r2_2[index_3d(snk_3_b[0],wnum,snk_3_nq[0] ,Nw2,Nq)];
         b_snk_color_weights(1, 0, wnum, nperm, 4) = snk_color_weights_r2_2[index_3d(snk_1_b[1],wnum,snk_1_nq[1] ,Nw2,Nq)];
         b_snk_spin_weights(1, 0, wnum, nperm, 4) = snk_spin_weights_r2_2[index_3d(snk_1_b[1],wnum,snk_1_nq[1] ,Nw2,Nq)];
         b_snk_color_weights(1, 1, wnum, nperm, 4) = snk_color_weights_r2_2[index_3d(snk_2_b[1],wnum,snk_2_nq[1] ,Nw2,Nq)];
         b_snk_spin_weights(1, 1, wnum, nperm, 4) = snk_spin_weights_r2_2[index_3d(snk_2_b[1],wnum,snk_2_nq[1] ,Nw2,Nq)];
         b_snk_color_weights(1, 2, wnum, nperm, 4) = snk_color_weights_r2_2[index_3d(snk_3_b[1],wnum,snk_3_nq[1] ,Nw2,Nq)];
         b_snk_spin_weights(1, 2, wnum, nperm, 4) = snk_spin_weights_r2_2[index_3d(snk_3_b[1],wnum,snk_3_nq[1] ,Nw2,Nq)]; 

         b_snk_color_weights(0, 0, wnum, nperm, 5) = snk_color_weights_r2_2[index_3d(snk_1_b[0],wnum,snk_1_nq[0] ,Nw2,Nq)];
         b_snk_spin_weights(0, 0, wnum, nperm, 5) = snk_spin_weights_r2_2[index_3d(snk_1_b[0],wnum,snk_1_nq[0] ,Nw2,Nq)];
         b_snk_color_weights(0, 1, wnum, nperm, 5) = snk_color_weights_r2_2[index_3d(snk_2_b[0],wnum,snk_2_nq[0] ,Nw2,Nq)];
         b_snk_spin_weights(0, 1, wnum, nperm, 5) = snk_spin_weights_r2_2[index_3d(snk_2_b[0],wnum,snk_2_nq[0] ,Nw2,Nq)];
         b_snk_color_weights(0, 2, wnum, nperm, 5) = snk_color_weights_r2_2[index_3d(snk_3_b[0],wnum,snk_3_nq[0] ,Nw2,Nq)];
         b_snk_spin_weights(0, 2, wnum, nperm, 5) = snk_spin_weights_r2_2[index_3d(snk_3_b[0],wnum,snk_3_nq[0] ,Nw2,Nq)];
         b_snk_color_weights(1, 0, wnum, nperm, 5) = snk_color_weights_r2_2[index_3d(snk_1_b[1],wnum,snk_1_nq[1] ,Nw2,Nq)];
         b_snk_spin_weights(1, 0, wnum, nperm, 5) = snk_spin_weights_r2_2[index_3d(snk_1_b[1],wnum,snk_1_nq[1] ,Nw2,Nq)];
         b_snk_color_weights(1, 1, wnum, nperm, 5) = snk_color_weights_r2_2[index_3d(snk_2_b[1],wnum,snk_2_nq[1] ,Nw2,Nq)];
         b_snk_spin_weights(1, 1, wnum, nperm, 5) = snk_spin_weights_r2_2[index_3d(snk_2_b[1],wnum,snk_2_nq[1] ,Nw2,Nq)];
         b_snk_color_weights(1, 2, wnum, nperm, 5) = snk_color_weights_r2_2[index_3d(snk_3_b[1],wnum,snk_3_nq[1] ,Nw2,Nq)];
         b_snk_spin_weights(1, 2, wnum, nperm, 5) = snk_spin_weights_r2_2[index_3d(snk_3_b[1],wnum,snk_3_nq[1] ,Nw2,Nq)]; 
      }
      for (int wnum=0; wnum< Nw2Hex; wnum++) {
         for (int q=0; q < 2*Nq; q++) {
            b_hex_snk_color_weights((q-(q%Nq))/Nq, q%Nq, wnum, nperm, 0) = hex_snk_color_weights_T1_r1[index_2d(wnum,perms[index_2d(nperm,q ,2*Nq)]-1 ,2*Nq)];
            b_hex_snk_spin_weights((q-(q%Nq))/Nq, q%Nq, wnum, nperm, 0) = hex_snk_spin_weights_T1_r1[index_2d(wnum,perms[index_2d(nperm,q ,2*Nq)]-1 ,2*Nq)];
            b_hex_snk_color_weights((q-(q%Nq))/Nq, q%Nq, wnum, nperm, 1) = hex_snk_color_weights_T1_r3[index_2d(wnum,perms[index_2d(nperm,q ,2*Nq)]-1 ,2*Nq)];
            b_hex_snk_spin_weights((q-(q%Nq))/Nq, q%Nq, wnum, nperm, 1) = hex_snk_spin_weights_T1_r3[index_2d(wnum,perms[index_2d(nperm,q ,2*Nq)]-1 ,2*Nq)];
            b_hex_snk_color_weights((q-(q%Nq))/Nq, q%Nq, wnum, nperm, 2) = hex_snk_color_weights_T1_r2[index_2d(wnum,perms[index_2d(nperm,q ,2*Nq)]-1 ,2*Nq)];
            b_hex_snk_spin_weights((q-(q%Nq))/Nq, q%Nq, wnum, nperm, 2) = hex_snk_spin_weights_T1_r2[index_2d(wnum,perms[index_2d(nperm,q ,2*Nq)]-1 ,2*Nq)];
            b_hex_snk_color_weights((q-(q%Nq))/Nq, q%Nq, wnum, nperm, 3) = hex_snk_color_weights_T1_r2[index_2d(wnum,perms[index_2d(nperm,q ,2*Nq)]-1 ,2*Nq)];
            b_hex_snk_spin_weights((q-(q%Nq))/Nq, q%Nq, wnum, nperm, 3) = hex_snk_spin_weights_T1_r2[index_2d(wnum,perms[index_2d(nperm,q ,2*Nq)]-1 ,2*Nq)];
            b_hex_snk_color_weights((q-(q%Nq))/Nq, q%Nq, wnum, nperm, 4) = hex_snk_color_weights_A1[index_2d(wnum,perms[index_2d(nperm,q ,2*Nq)]-1 ,2*Nq)];
            b_hex_snk_spin_weights((q-(q%Nq))/Nq, q%Nq, wnum, nperm, 4) = hex_snk_spin_weights_A1[index_2d(wnum,perms[index_2d(nperm,q ,2*Nq)]-1 ,2*Nq)];
            b_hex_snk_color_weights((q-(q%Nq))/Nq, q%Nq, wnum, nperm, 5) = hex_snk_color_weights_A1[index_2d(wnum,perms[index_2d(nperm,q ,2*Nq)]-1 ,2*Nq)];
            b_hex_snk_spin_weights((q-(q%Nq))/Nq, q%Nq, wnum, nperm, 5) = hex_snk_spin_weights_A1[index_2d(wnum,perms[index_2d(nperm,q ,2*Nq)]-1 ,2*Nq)];
            //printf("hex color weight (%d, %d) r=4: %d \n", wnum, q, b_hex_snk_color_weights((q-(q%Nq))/Nq, q%Nq, wnum, nperm, 4));
            //printf("hex color weight (%d, %d) r=4: %d \n", wnum, q, b_hex_snk_color_weights((q-(q%Nq))/Nq, q%Nq, wnum, nperm, 4));
            //printf("hex spin weight (%d, %d) r=4: %d \n", wnum, q, b_hex_snk_spin_weights((q-(q%Nq))/Nq, q%Nq, wnum, nperm, 4));
         }
      }
   }
   for (int wnum=0; wnum< Nw2Hex; wnum++) {
      b_hex_snk_weights(wnum, 0) = hex_snk_weights_T1_r1[wnum];
      b_hex_snk_weights(wnum, 1) = hex_snk_weights_T1_r3[wnum];
      b_hex_snk_weights(wnum, 2) = hex_snk_weights_T1_r2[wnum];
      b_hex_snk_weights(wnum, 3) = hex_snk_weights_T1_r2[wnum];
      b_hex_snk_weights(wnum, 4) = hex_snk_weights_A1[wnum];
      b_hex_snk_weights(wnum, 5) = hex_snk_weights_A1[wnum];
      //printf("hex weight %d r=4: %4.9f \n", wnum, b_hex_snk_weights(wnum, 4));
   }

   for (int b=0; b<Nr; b++)
      for (int m=0; m<Nsrc+NsrcHex; m++)
         for (int n=0; n<Nsnk+NsnkHex; n++)
            for (int t=0; t<Lt; t++) //{
               //b_C_r(n,b,m,t) = 0.0;
               //b_C_i(n,b,m,t) = 0.0;
               for (int x=0; x<Vsnk; x++) {
                  b_C_r(n,b,m,x,t) = 0.0;
                  b_C_i(n,b,m,x,t) = 0.0;
            } 

   if (rank == 0) {
   printf("prop 1 %4.16f + I %4.16f \n", b_B1_prop_r(0,0,0,0,0,0,0,0), b_B1_prop_i(0,0,0,0,0,0,0,0));
   printf("prop 2 %4.16f + I %4.16f \n", b_B2_prop_r(0,0,0,0,0,0,0,0), b_B2_prop_i(0,0,0,0,0,0,0,0));
   printf("psi src 1 %4.16f + I %4.16f \n", b_B1_src_psi_r(0,0), b_B1_src_psi_i(0,0));
   printf("psi src 2 %4.16f + I %4.16f \n", b_B2_src_psi_r(0,0), b_B2_src_psi_i(0,0));
   printf("psi snk %4.16f + I %4.16f \n", b_snk_psi_r(0,0,0), b_snk_psi_i(0,0,0));
   printf("weights snk %4.9e \n", b_snk_weights(0,0));
   printf("sigs %d \n", b_sigs(0));
   }
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
                b_B1_snk_psi_r.raw_buffer(),
                b_B1_snk_psi_i.raw_buffer(),
                b_B2_snk_psi_r.raw_buffer(),
                b_B2_snk_psi_i.raw_buffer(),
                b_hex_src_psi_r.raw_buffer(),
                b_hex_src_psi_i.raw_buffer(),
                b_hex_snk_psi_r.raw_buffer(),
                b_hex_snk_psi_i.raw_buffer(),
                b_snk_psi_r.raw_buffer(),
                b_snk_psi_i.raw_buffer(),
				    b_snk_blocks.raw_buffer(),
				    b_sigs.raw_buffer(),
				 b_src_color_weights.raw_buffer(),
				 b_src_spin_weights.raw_buffer(),
				 b_src_weights.raw_buffer(),
				    b_snk_b.raw_buffer(),
				    b_snk_color_weights.raw_buffer(),
				    b_snk_spin_weights.raw_buffer(),
				    b_snk_weights.raw_buffer(),
                                    b_hex_snk_color_weights.raw_buffer(),
                                    b_hex_snk_spin_weights.raw_buffer(),
                                    b_hex_snk_weights.raw_buffer());

   if (rank == 0) {
   /*printf("BB-BB non-zero? %4.9e + I %4.9e \n", b_C_r(0,0,0,0), b_C_i(0,0,0,0) );
   printf("BB-BB non-zero? %4.9e + I %4.9e \n", b_C_r(0,1,0,0), b_C_i(0,1,0,0) );
   printf("BB-BB non-zero? %4.9e + I %4.9e \n", b_C_r(0,2,0,0), b_C_i(0,2,0,0) );
   printf("BB-BB non-zero? %4.9e + I %4.9e \n", b_C_r(0,3,0,0), b_C_i(0,3,0,0) );
   printf("BB-BB non-zero? %4.9e + I %4.9e \n", b_C_r(0,4,0,0), b_C_i(0,4,0,0) );
   printf("BB-BB non-zero? %4.9e + I %4.9e \n", b_C_r(0,5,0,0), b_C_i(0,5,0,0) );
   printf("H-BB non-zero? %4.9e + I %4.9e \n", b_C_r(0,0,Nsrc,0), b_C_i(0,0,Nsrc,0) );
   printf("H-BB non-zero? %4.9e + I %4.9e \n", b_C_r(0,1,Nsrc,0), b_C_i(0,1,Nsrc,0) );
   printf("H-BB non-zero? %4.9e + I %4.9e \n", b_C_r(0,2,Nsrc,0), b_C_i(0,2,Nsrc,0) );
   printf("H-BB non-zero? %4.9e + I %4.9e \n", b_C_r(0,3,Nsrc,0), b_C_i(0,3,Nsrc,0) );
   printf("H-BB non-zero? %4.9e + I %4.9e \n", b_C_r(0,4,Nsrc,0), b_C_i(0,4,Nsrc,0) );
   printf("H-BB non-zero? %4.9e + I %4.9e \n", b_C_r(0,5,Nsrc,0), b_C_i(0,5,Nsrc,0) );
   printf("H-H non-zero? %4.9e + I %4.9e \n", b_C_r(Nsnk,0,Nsrc,0), b_C_i(Nsnk,0,Nsrc,0) );
   printf("H-H non-zero? %4.9e + I %4.9e \n", b_C_r(Nsnk,1,Nsrc,0), b_C_i(Nsnk,1,Nsrc,0) );
   printf("H-H non-zero? %4.9e + I %4.9e \n", b_C_r(Nsnk,2,Nsrc,0), b_C_i(Nsnk,2,Nsrc,0) );
   printf("H-H non-zero? %4.9e + I %4.9e \n", b_C_r(Nsnk,3,Nsrc,0), b_C_i(Nsnk,3,Nsrc,0) );
   printf("H-H non-zero? %4.9e + I %4.9e \n", b_C_r(Nsnk,4,Nsrc,0), b_C_i(Nsnk,4,Nsrc,0) );
   printf("H-H non-zero? %4.9e + I %4.9e \n", b_C_r(Nsnk,5,Nsrc,0), b_C_i(Nsnk,5,Nsrc,0) ); */
   printf("BB-BB non-zero? %4.9e + I %4.9e \n", b_C_r(0,0,0,0,0), b_C_i(0,0,0,0,0) );
   printf("BB-BB non-zero? %4.9e + I %4.9e \n", b_C_r(0,1,0,0,0), b_C_i(0,1,0,0,0) );
   printf("BB-BB non-zero? %4.9e + I %4.9e \n", b_C_r(0,2,0,0,0), b_C_i(0,2,0,0,0) );
   printf("BB-BB non-zero? %4.9e + I %4.9e \n", b_C_r(0,3,0,0,0), b_C_i(0,3,0,0,0) );
   printf("BB-BB non-zero? %4.9e + I %4.9e \n", b_C_r(0,4,0,0,0), b_C_i(0,4,0,0,0) );
   printf("BB-BB non-zero? %4.9e + I %4.9e \n", b_C_r(0,5,0,0,0), b_C_i(0,5,0,0,0) );
   printf("H-BB non-zero? %4.9e + I %4.9e \n", b_C_r(0,0,Nsrc,0,0), b_C_i(0,0,Nsrc,0,0) );
   printf("H-BB non-zero? %4.9e + I %4.9e \n", b_C_r(0,1,Nsrc,0,0), b_C_i(0,1,Nsrc,0,0) );
   printf("H-BB non-zero? %4.9e + I %4.9e \n", b_C_r(0,2,Nsrc,0,0), b_C_i(0,2,Nsrc,0,0) );
   printf("H-BB non-zero? %4.9e + I %4.9e \n", b_C_r(0,3,Nsrc,0,0), b_C_i(0,3,Nsrc,0,0) );
   printf("H-BB non-zero? %4.9e + I %4.9e \n", b_C_r(0,4,Nsrc,0,0), b_C_i(0,4,Nsrc,0,0) );
   printf("H-BB non-zero? %4.9e + I %4.9e \n", b_C_r(0,5,Nsrc,0,0), b_C_i(0,5,Nsrc,0,0) );
   printf("H-H non-zero? %4.9e + I %4.9e \n", b_C_r(Nsnk,0,Nsrc,0,0), b_C_i(Nsnk,0,Nsrc,0,0) );
   printf("H-H non-zero? %4.9e + I %4.9e \n", b_C_r(Nsnk,1,Nsrc,0,0), b_C_i(Nsnk,1,Nsrc,0,0) );
   printf("H-H non-zero? %4.9e + I %4.9e \n", b_C_r(Nsnk,2,Nsrc,0,0), b_C_i(Nsnk,2,Nsrc,0,0) );
   printf("H-H non-zero? %4.9e + I %4.9e \n", b_C_r(Nsnk,3,Nsrc,0,0), b_C_i(Nsnk,3,Nsrc,0,0) );
   printf("H-H non-zero? %4.9e + I %4.9e \n", b_C_r(Nsnk,4,Nsrc,0,0), b_C_i(Nsnk,4,Nsrc,0,0) );
   printf("H-H non-zero? %4.9e + I %4.9e \n", b_C_r(Nsnk,5,Nsrc,0,0), b_C_i(Nsnk,5,Nsrc,0,0) ); 
   }

    // symmetrize and such
      for (int m=0; m<Nsrc; m++)
         for (int n=0; n<Nsnk; n++)
            for (int t=0; t<Lt; t++) {
                  double number0r;
                  double number0i;
                  double number1r;
                  double number1i;
                  double number2r;
                  double number2i;
                  double number3r;
                  double number3i;
                  double number4r;
                  double number4i;
                  double number5r;
                  double number5i;
                  double this_number0r = b_C_r(n,0,m,rank,t);
                  double this_number0i = b_C_i(n,0,m,rank,t);
                  double this_number1r = b_C_r(n,1,m,rank,t);
                  double this_number1i = b_C_i(n,1,m,rank,t);
                  double this_number2r = b_C_r(n,2,m,rank,t);
                  double this_number2i = b_C_i(n,2,m,rank,t);
                  double this_number3r = b_C_r(n,3,m,rank,t);
                  double this_number3i = b_C_i(n,3,m,rank,t);
                  double this_number4r = b_C_r(n,4,m,rank,t);
                  double this_number4i = b_C_i(n,4,m,rank,t);
                  double this_number5r = b_C_r(n,5,m,rank,t);
                  double this_number5i = b_C_i(n,5,m,rank,t);
               MPI_Allreduce(&this_number0r, &number0r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
               MPI_Allreduce(&this_number0i, &number0i, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
               MPI_Allreduce(&this_number1r, &number1r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
               MPI_Allreduce(&this_number1i, &number1i, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
               MPI_Allreduce(&this_number2r, &number2r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
               MPI_Allreduce(&this_number2i, &number2i, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
               MPI_Allreduce(&this_number3r, &number3r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
               MPI_Allreduce(&this_number3i, &number3i, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
               MPI_Allreduce(&this_number4r, &number4r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
               MPI_Allreduce(&this_number4i, &number4i, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
               MPI_Allreduce(&this_number5r, &number5r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
               MPI_Allreduce(&this_number5i, &number5i, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
               MPI_Barrier(MPI_COMM_WORLD);
                 C_re[index_4d(0,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Lt)] = 1/sqrt(2) * (number2r - number3r - number4r + number5r);
                 C_re[index_4d(1,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Lt)] = number0r;
                 C_re[index_4d(2,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Lt)] = 1/sqrt(2) * (number2r + number3r + number4r + number5r);
                 C_re[index_4d(3,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Lt)] = number1r;
                 C_im[index_4d(0,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Lt)] = 1/sqrt(2) * (number2i - number3i - number4i + number5i);
                 C_im[index_4d(1,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Lt)] = number0i;
                 C_im[index_4d(2,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Lt)] = 1/sqrt(2) * (number2i + number3i + number4i + number5i);
                 C_im[index_4d(3,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Lt)] = number1i;
            }
       for (int m=Nsrc; m<Nsrc+NsrcHex; m++)
         for (int n=0; n<Nsnk; n++)
            for (int t=0; t<Lt; t++) {
                  double number0r;
                  double number0i;
                  double number1r;
                  double number1i;
                  double number2r;
                  double number2i;
                  double number3r;
                  double number3i;
                  double number4r;
                  double number4i;
                  double number5r;
                  double number5i;
                  double this_number0r = b_C_r(n,0,m,rank,t);
                  double this_number0i = b_C_i(n,0,m,rank,t);
                  double this_number1r = b_C_r(n,1,m,rank,t);
                  double this_number1i = b_C_i(n,1,m,rank,t);
                  double this_number2r = b_C_r(n,2,m,rank,t);
                  double this_number2i = b_C_i(n,2,m,rank,t);
                  double this_number3r = b_C_r(n,3,m,rank,t);
                  double this_number3i = b_C_i(n,3,m,rank,t);
                  double this_number4r = b_C_r(n,4,m,rank,t);
                  double this_number4i = b_C_i(n,4,m,rank,t);
                  double this_number5r = b_C_r(n,5,m,rank,t);
                  double this_number5i = b_C_i(n,5,m,rank,t);
               MPI_Allreduce(&this_number0r, &number0r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
               MPI_Allreduce(&this_number0i, &number0i, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
               MPI_Allreduce(&this_number1r, &number1r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
               MPI_Allreduce(&this_number1i, &number1i, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
               MPI_Allreduce(&this_number2r, &number2r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
               MPI_Allreduce(&this_number2i, &number2i, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
               MPI_Allreduce(&this_number3r, &number3r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
               MPI_Allreduce(&this_number3i, &number3i, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
               MPI_Allreduce(&this_number4r, &number4r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
               MPI_Allreduce(&this_number4i, &number4i, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
               MPI_Allreduce(&this_number5r, &number5r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
               MPI_Allreduce(&this_number5i, &number5i, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
               MPI_Barrier(MPI_COMM_WORLD);
                 C_re[index_4d(0,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Lt)] = 1/sqrt(2) * (number4r - number5r);
                 C_re[index_4d(1,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Lt)] = -number0r;
                 C_re[index_4d(2,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Lt)] = 1/sqrt(2) * (number2r + number3r);
                 C_re[index_4d(3,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Lt)] = -number1r;
                 C_im[index_4d(0,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Lt)] = 1/sqrt(2) * (number4i - number5i);
                 C_im[index_4d(1,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Lt)] = -number0i;
                 C_im[index_4d(2,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Lt)] = 1/sqrt(2) * (number2i + number3i);
                 C_im[index_4d(3,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Lt)] = -number1i;
            }
      for (int m=0; m<Nsrc; m++)
         for (int n=Nsnk; n<Nsnk+NsnkHex; n++)
            for (int t=0; t<Lt; t++) {
                  double number0r;
                  double number0i;
                  double number1r;
                  double number1i;
                  double number2r;
                  double number2i;
                  double number3r;
                  double number3i;
                  double number4r;
                  double number4i;
                  double number5r;
                  double number5i;
                  double this_number0r = b_C_r(n,0,m,rank,t);
                  double this_number0i = b_C_i(n,0,m,rank,t);
                  double this_number1r = b_C_r(n,1,m,rank,t);
                  double this_number1i = b_C_i(n,1,m,rank,t);
                  double this_number2r = b_C_r(n,2,m,rank,t);
                  double this_number2i = b_C_i(n,2,m,rank,t);
                  double this_number3r = b_C_r(n,3,m,rank,t);
                  double this_number3i = b_C_i(n,3,m,rank,t);
                  double this_number4r = b_C_r(n,4,m,rank,t);
                  double this_number4i = b_C_i(n,4,m,rank,t);
                  double this_number5r = b_C_r(n,5,m,rank,t);
                  double this_number5i = b_C_i(n,5,m,rank,t);
               MPI_Allreduce(&this_number0r, &number0r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
               MPI_Allreduce(&this_number0i, &number0i, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
               MPI_Allreduce(&this_number1r, &number1r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
               MPI_Allreduce(&this_number1i, &number1i, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
               MPI_Allreduce(&this_number2r, &number2r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
               MPI_Allreduce(&this_number2i, &number2i, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
               MPI_Allreduce(&this_number3r, &number3r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
               MPI_Allreduce(&this_number3i, &number3i, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
               MPI_Allreduce(&this_number4r, &number4r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
               MPI_Allreduce(&this_number4i, &number4i, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
               MPI_Allreduce(&this_number5r, &number5r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
               MPI_Allreduce(&this_number5i, &number5i, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
               MPI_Barrier(MPI_COMM_WORLD);
                 C_re[index_4d(0,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Lt)] = 1/sqrt(2) * (number4r - number5r);
                 C_re[index_4d(1,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Lt)] = -number0r;
                 C_re[index_4d(2,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Lt)] = 1/sqrt(2) * (number2r + number3r);
                 C_re[index_4d(3,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Lt)] = -number1r;
                 C_im[index_4d(0,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Lt)] = 1/sqrt(2) * (number4i - number5i);
                 C_im[index_4d(1,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Lt)] = -number0i;
                 C_im[index_4d(2,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Lt)] = 1/sqrt(2) * (number2i + number3i);
                 C_im[index_4d(3,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Lt)] = -number1i;
            }
      for (int m=Nsrc; m<Nsrc+NsrcHex; m++)
         for (int n=Nsnk; n<Nsnk+NsnkHex; n++)
            for (int t=0; t<Lt; t++) {
                  double number0r;
                  double number0i;
                  double number1r;
                  double number1i;
                  double number2r;
                  double number2i;
                  double number4r;
                  double number4i;
                  double this_number0r = b_C_r(n,0,m,rank,t);
                  double this_number0i = b_C_i(n,0,m,rank,t);
                  double this_number1r = b_C_r(n,1,m,rank,t);
                  double this_number1i = b_C_i(n,1,m,rank,t);
                  double this_number2r = b_C_r(n,2,m,rank,t);
                  double this_number2i = b_C_i(n,2,m,rank,t);
                  double this_number4r = b_C_r(n,4,m,rank,t);
                  double this_number4i = b_C_i(n,4,m,rank,t);
               MPI_Allreduce(&this_number0r, &number0r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
               MPI_Allreduce(&this_number0i, &number0i, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
               MPI_Allreduce(&this_number1r, &number1r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
               MPI_Allreduce(&this_number1i, &number1i, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
               MPI_Allreduce(&this_number2r, &number2r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
               MPI_Allreduce(&this_number2i, &number2i, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
               MPI_Allreduce(&this_number4r, &number4r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
               MPI_Allreduce(&this_number4i, &number4i, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
               MPI_Barrier(MPI_COMM_WORLD);
                 C_re[index_4d(0,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Lt)] = number4r;
                 C_re[index_4d(1,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Lt)] = number0r;
                 C_re[index_4d(2,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Lt)] = number2r;
                 C_re[index_4d(3,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Lt)] = number1r;
                 C_im[index_4d(0,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Lt)] = number4i;
                 C_im[index_4d(1,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Lt)] = number0i;
                 C_im[index_4d(2,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Lt)] = number2i;
                 C_im[index_4d(3,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Lt)] = number1i;
            } 

   if (rank == 0) {
   for (int b=0; b<4; b++) {
      printf("\n");
      for (int m=0; m<Nsrc+NsrcHex; m++)
         for (int n=0; n<Nsnk+NsnkHex; n++)
            for (int t=0; t<Lt; t++) {
                  printf("r=%d, m=%d, n=%d, t=%d: %4.9e + I (%4.9e) \n", b, m, n, t, C_re[index_4d(b,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Lt)],  C_im[index_4d(b,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Lt)]);
            }
   }
   }

   free(snk_color_weights_1);
   free(snk_color_weights_2);
   free(snk_color_weights_r1);
   free(snk_color_weights_r2_1);
   free(snk_color_weights_r2_2);
   free(snk_color_weights_r3);
   free(snk_spin_weights_1);
   free(snk_spin_weights_2);
   free(snk_spin_weights_r1);
   free(snk_spin_weights_r2_1);
   free(snk_spin_weights_r2_2);
   free(snk_spin_weights_r3);
}

#ifdef __cplusplus
}  // extern "C"
#endif
