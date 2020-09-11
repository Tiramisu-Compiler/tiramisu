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

#include "tiramisu_make_fused_dibaryon_blocks_correlator_wrapper.h"
#include "tiramisu_make_fused_dibaryon_blocks_correlator_ref.cpp"

#define RUN_REFERENCE 1
#define RUN_CHECK 1
int nb_tests = 1;
int randommode = 0;



void tiramisu_make_two_nucleon_2pt(double* C_re,
    double* C_im,
     double* B1_prop_re, 
     double* B1_prop_im, 
     double* B2_prop_re, 
     double* B2_prop_im, 
     int *src_color_weights_r1,
     int *src_spin_weights_r1,
     double *src_weights_r1,
     int *src_color_weights_r2,
     int *src_spin_weights_r2,
     double *src_weights_r2,
     int *hex_snk_color_weights_A1,
     int *hex_snk_spin_weights_A1,
     double *hex_snk_weights_A1,
     int *hex_snk_color_weights_T1_r1,
     int *hex_snk_spin_weights_T1_r1,
     double *hex_snk_weights_T1_r1,
     int *hex_snk_color_weights_T1_r2,
     int *hex_snk_spin_weights_T1_r2,
     double *hex_snk_weights_T1_r2,
     int *hex_snk_color_weights_T1_r3,
     int *hex_snk_spin_weights_T1_r3,
     double *hex_snk_weights_T1_r3,
     int* perms, 
     int* sigs, 
     double* src_psi_B1_re, 
     double* src_psi_B1_im, 
     double* src_psi_B2_re, 
     double* src_psi_B2_im, 
     double* snk_psi_re,
     double* snk_psi_im, 
     double* snk_psi_B1_re, 
     double* snk_psi_B1_im, 
     double* snk_psi_B2_re, 
     double* snk_psi_B2_im, 
     double* hex_src_psi_re, 
     double* hex_src_psi_im, 
     double* hex_snk_psi_re, 
     double* hex_snk_psi_im,
     int space_symmetric,
     int snk_entangled)
{
   int q, t, iC, iS, jC, jS, y, x, x1, x2, m, n, k, wnum, nperm, b, r, rp;
   int iC1, iS1, iC2, iS2, jC1, jS1, jC2, jS2, kC1, kS1, kC2, kS2;

    int rank = 0;
#ifdef WITH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

   if (rank == 0) {
    long mega = 1024*1024;
    std::cout << "Array sizes" << std::endl;
    std::cout << "Prop:" <<  std::endl;
    std::cout << "	Max index size = " << Nq*Vsnk*Vsrc*Nc*Ns*Nc*Ns*Lt <<  std::endl;
    std::cout << "	Array size = " << Nq*Vsnk*Vsrc*Nc*Ns*Nc*Ns*Lt*sizeof(std::complex<double>)/mega << " Mega bytes" << std::endl;
    std::cout << "Q, O & P:" <<  std::endl;
    std::cout << "	Max index size = " << Vsnk*Vsrc*Nc*Ns*Nc*Ns <<  std::endl;
    std::cout << "	Array size = " << Vsnk*Vsrc*Nc*Ns*Nc*Ns*sizeof(std::complex<double>)/mega << " Mega bytes" <<  std::endl;

    long kilo = 1024;
    std::cout << "Blocal:" <<  std::endl;
    std::cout << "	Max index size = " << Vsnk*Nsrc*Nc*Ns*Nc*Ns*Nc*Ns <<  std::endl;
    std::cout << "	Array size = " << Vsnk*Nsrc*Nc*Ns*Nc*Ns*Nc*Ns*sizeof(std::complex<double>)/kilo << " kilo bytes" <<  std::endl;
    std::cout << "Blocal, Bsingle, Bdouble:" <<  std::endl;
    std::cout << "	Max index size = " << Nc*Ns*Nc*Ns*Nc*Ns <<  std::endl;
    std::cout << "	Array size = " << Nc*Ns*Nc*Ns*Nc*Ns*sizeof(std::complex<double>)/kilo << " kilo bytes" <<  std::endl;
    std::cout << std::endl;
   }

   // Halide buffers
   Halide::Buffer<double> b_C_r(Nsnk+NsnkHex, B2Nrows, Nsrc+NsrcHex, B2Nrows, Vsnk/sites_per_rank, Lt, "C_r");
   Halide::Buffer<double> b_C_i(Nsnk+NsnkHex, B2Nrows, Nsrc+NsrcHex, B2Nrows, Vsnk/sites_per_rank, Lt, "C_i");

   Halide::Buffer<int> b_src_color_weights(Nq, Nw, B2Nrows, "src_color_weights");
   Halide::Buffer<int> b_src_spin_weights(Nq, Nw, B2Nrows, "src_spin_weights");
   Halide::Buffer<double> b_src_weights(Nw, B2Nrows, "src_weights");

   Halide::Buffer<int> b_src_spins(2, 2, B2Nrows, "src_spins");
   Halide::Buffer<double> b_src_spin_block_weights(2, B2Nrows, "src_spin_block_weights");
   Halide::Buffer<int> b_snk_b(2, Nq, Nperms, "snk_b");
   Halide::Buffer<int> b_snk_color_weights(2, Nq, Nw2, Nperms, B2Nrows, "snk_color_weights");
   Halide::Buffer<int> b_snk_spin_weights(2, Nq, Nw2, Nperms, B2Nrows, "snk_spin_weights");
   Halide::Buffer<double> b_snk_weights(Nw2, B2Nrows, "snk_weights");
   Halide::Buffer<int> b_hex_snk_color_weights(2, Nq, Nw2Hex, Nperms, B2Nrows, "hex_snk_color_weights");
   Halide::Buffer<int> b_hex_snk_spin_weights(2, Nq, Nw2Hex, Nperms, B2Nrows, "hex_snk_spin_weights");
   Halide::Buffer<double> b_hex_snk_weights(Nw2Hex, B2Nrows, "hex_snk_weights");

    // prop
    Halide::Buffer<double> b_B1_prop_r((double *)B1_prop_re, {Vsrc, Vsnk, Ns, Nc, Ns, Nc, Lt, Nq});
    Halide::Buffer<double> b_B1_prop_i((double *)B1_prop_im, {Vsrc, Vsnk, Ns, Nc, Ns, Nc, Lt, Nq});
    Halide::Buffer<double> b_B2_prop_r((double *)B2_prop_re, {Vsrc, Vsnk, Ns, Nc, Ns, Nc, Lt, Nq});
    Halide::Buffer<double> b_B2_prop_i((double *)B2_prop_im, {Vsrc, Vsnk, Ns, Nc, Ns, Nc, Lt, Nq});

   if (rank == 0) {
   printf("prop elem %4.9f \n", b_B1_prop_r(0,0,0,0,0,0,0,0));
   }

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
    Halide::Buffer<double> b_snk_psi_r((double *)snk_psi_re, {Nsnk, Vsnk, Vsnk});
    Halide::Buffer<double> b_snk_psi_i((double *)snk_psi_im, {Nsnk, Vsnk, Vsnk});


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
   int* snk_color_weights_r1 = (int *) malloc(2 * Nw2 * Nq * sizeof (int));
   int* snk_color_weights_r2 = (int *) malloc(2 * Nw2 * Nq * sizeof (int));
   int* snk_color_weights_r3 = (int *) malloc(2 * Nw2 * Nq * sizeof (int));
   int* snk_spin_weights_r1 = (int *) malloc(2 * Nw2 * Nq * sizeof (int));
   int* snk_spin_weights_r2 = (int *) malloc(2 * Nw2 * Nq * sizeof (int));
   int* snk_spin_weights_r3 = (int *) malloc(2 * Nw2 * Nq * sizeof (int));
   for (int nB1=0; nB1<Nw; nB1++) {
      for (int nB2=0; nB2<Nw; nB2++) {
         b_snk_weights(nB1+Nw*nB2, 0) = 1.0/sqrt(2) * src_weights_r1[nB1]*src_weights_r2[nB2];
         b_snk_weights(nB1+Nw*nB2, 1) = src_weights_r1[nB1]*src_weights_r1[nB2];
         b_snk_weights(nB1+Nw*nB2, 2) = 1.0/sqrt(2) * src_weights_r1[nB1]*src_weights_r2[nB2];
         b_snk_weights(nB1+Nw*nB2, 3) = src_weights_r2[nB1]*src_weights_r2[nB2];
         for (int nq=0; nq<Nq; nq++) {
            // T1g_r1
            snk_color_weights_r1[index_3d(0,nB1+Nw*nB2,nq ,Nw2,Nq)] = src_color_weights_r1[index_2d(nB1,nq ,Nq)];
            snk_spin_weights_r1[index_3d(0,nB1+Nw*nB2,nq ,Nw2,Nq)] = src_spin_weights_r1[index_2d(nB1,nq ,Nq)];
            snk_color_weights_r1[index_3d(1,nB1+Nw*nB2,nq ,Nw2,Nq)] = src_color_weights_r1[index_2d(nB2,nq ,Nq)];
            snk_spin_weights_r1[index_3d(1,nB1+Nw*nB2,nq ,Nw2,Nq)] = src_spin_weights_r1[index_2d(nB2,nq ,Nq)];
            // T1g_r2 (and A1g)
            snk_color_weights_r2[index_3d(0,nB1+Nw*nB2,nq ,Nw2,Nq)] = src_color_weights_r1[index_2d(nB1,nq ,Nq)];
            snk_spin_weights_r2[index_3d(0,nB1+Nw*nB2,nq ,Nw2,Nq)] = src_spin_weights_r1[index_2d(nB1,nq ,Nq)];
            snk_color_weights_r2[index_3d(1,nB1+Nw*nB2,nq ,Nw2,Nq)] = src_color_weights_r2[index_2d(nB2,nq ,Nq)];
            snk_spin_weights_r2[index_3d(1,nB1+Nw*nB2,nq ,Nw2,Nq)] = src_spin_weights_r2[index_2d(nB2,nq ,Nq)];
            // T1g_r3 
            snk_color_weights_r3[index_3d(0,nB1+Nw*nB2,nq ,Nw2,Nq)] = src_color_weights_r2[index_2d(nB1,nq ,Nq)];
            snk_spin_weights_r3[index_3d(0,nB1+Nw*nB2,nq ,Nw2,Nq)] = src_spin_weights_r2[index_2d(nB1,nq ,Nq)];
            snk_color_weights_r3[index_3d(1,nB1+Nw*nB2,nq ,Nw2,Nq)] = src_color_weights_r2[index_2d(nB2,nq ,Nq)];
            snk_spin_weights_r3[index_3d(1,nB1+Nw*nB2,nq ,Nw2,Nq)] = src_spin_weights_r2[index_2d(nB2,nq ,Nq)];
         }
      }
   }
   for (int nB1=0; nB1<Nw; nB1++) {
      for (int nB2=0; nB2<Nw; nB2++) {
         b_snk_weights(Nw*Nw+nB1+Nw*nB2, 0) = -1.0/sqrt(2) * src_weights_r2[nB1]*src_weights_r1[nB2];
         b_snk_weights(Nw*Nw+nB1+Nw*nB2, 1) = 0.0;
         b_snk_weights(Nw*Nw+nB1+Nw*nB2, 2) = 1.0/sqrt(2) * src_weights_r2[nB1]*src_weights_r1[nB2];
         b_snk_weights(Nw*Nw+nB1+Nw*nB2, 3) = 0.0;
         for (int nq=0; nq<Nq; nq++) {
           // T1g_r1
            snk_color_weights_r1[index_3d(0,Nw*Nw+nB1+Nw*nB2,nq ,Nw2,Nq)] = src_color_weights_r1[index_2d(nB1,nq ,Nq)];
            snk_spin_weights_r1[index_3d(0,Nw*Nw+nB1+Nw*nB2,nq ,Nw2,Nq)] = src_spin_weights_r1[index_2d(nB1,nq ,Nq)];
            snk_color_weights_r1[index_3d(1,Nw*Nw+nB1+Nw*nB2,nq ,Nw2,Nq)] = src_color_weights_r1[index_2d(nB2,nq ,Nq)];
            snk_spin_weights_r1[index_3d(1,Nw*Nw+nB1+Nw*nB2,nq ,Nw2,Nq)] = src_spin_weights_r1[index_2d(nB2,nq ,Nq)];
            // T1g_r2
            snk_color_weights_r2[index_3d(0,Nw*Nw+nB1+Nw*nB2,nq ,Nw2,Nq)] = src_color_weights_r2[index_2d(nB1,nq ,Nq)];
            snk_spin_weights_r2[index_3d(0,Nw*Nw+nB1+Nw*nB2,nq ,Nw2,Nq)] = src_spin_weights_r2[index_2d(nB1,nq ,Nq)];
            snk_color_weights_r2[index_3d(1,Nw*Nw+nB1+Nw*nB2,nq ,Nw2,Nq)] = src_color_weights_r1[index_2d(nB2,nq ,Nq)];
            snk_spin_weights_r2[index_3d(1,Nw*Nw+nB1+Nw*nB2,nq ,Nw2,Nq)] = src_spin_weights_r1[index_2d(nB2,nq ,Nq)];
            // T1g_r3 
            snk_color_weights_r3[index_3d(0,Nw*Nw+nB1+Nw*nB2,nq ,Nw2,Nq)] = src_color_weights_r2[index_2d(nB1,nq ,Nq)];
            snk_spin_weights_r3[index_3d(0,Nw*Nw+nB1+Nw*nB2,nq ,Nw2,Nq)] = src_spin_weights_r2[index_2d(nB1,nq ,Nq)];
            snk_color_weights_r3[index_3d(1,Nw*Nw+nB1+Nw*nB2,nq ,Nw2,Nq)] = src_color_weights_r2[index_2d(nB2,nq ,Nq)];
            snk_spin_weights_r3[index_3d(1,Nw*Nw+nB1+Nw*nB2,nq ,Nw2,Nq)] = src_spin_weights_r2[index_2d(nB2,nq ,Nq)];
         }
      }
   }
   b_src_spins(0,0,0) = 1;
   b_src_spins(1,0,0) = 2;
   b_src_spins(0,1,0) = 2;
   b_src_spins(1,1,0) = 1;
   b_src_spins(0,0,1) = 1;
   b_src_spins(1,0,1) = 1;
   b_src_spins(0,1,1) = 1;
   b_src_spins(1,1,1) = 1;
   b_src_spins(0,0,2) = 1;
   b_src_spins(1,0,2) = 2;
   b_src_spins(0,1,2) = 2;
   b_src_spins(1,1,2) = 1;
   b_src_spins(0,0,3) = 2;
   b_src_spins(1,0,3) = 2;
   b_src_spins(0,1,3) = 2;
   b_src_spins(1,1,3) = 2;
   b_src_spin_block_weights(0,0) = 1.0/sqrt(2);
   b_src_spin_block_weights(1,0) = -1.0/sqrt(2);
   b_src_spin_block_weights(0,1) = 1.0;
   b_src_spin_block_weights(1,1) = 0.0;
   b_src_spin_block_weights(0,2) = 1.0/sqrt(2);
   b_src_spin_block_weights(1,2) = 1.0/sqrt(2);
   b_src_spin_block_weights(0,3) = 1.0;
   b_src_spin_block_weights(1,3) = 0.0;
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
         b_snk_color_weights(0, 0, wnum, nperm, 0) = snk_color_weights_r2[index_3d(snk_1_b[0],wnum,snk_1_nq[0] ,Nw2,Nq)];
         b_snk_spin_weights(0, 0, wnum, nperm, 0) = snk_spin_weights_r2[index_3d(snk_1_b[0],wnum,snk_1_nq[0] ,Nw2,Nq)];
         b_snk_color_weights(0, 1, wnum, nperm, 0) = snk_color_weights_r2[index_3d(snk_2_b[0],wnum,snk_2_nq[0] ,Nw2,Nq)];
         b_snk_spin_weights(0, 1, wnum, nperm, 0) = snk_spin_weights_r2[index_3d(snk_2_b[0],wnum,snk_2_nq[0] ,Nw2,Nq)];
         b_snk_color_weights(0, 2, wnum, nperm, 0) = snk_color_weights_r2[index_3d(snk_3_b[0],wnum,snk_3_nq[0] ,Nw2,Nq)];
         b_snk_spin_weights(0, 2, wnum, nperm, 0) = snk_spin_weights_r2[index_3d(snk_3_b[0],wnum,snk_3_nq[0] ,Nw2,Nq)];
         b_snk_color_weights(1, 0, wnum, nperm, 0) = snk_color_weights_r2[index_3d(snk_1_b[1],wnum,snk_1_nq[1] ,Nw2,Nq)];
         b_snk_spin_weights(1, 0, wnum, nperm, 0) = snk_spin_weights_r2[index_3d(snk_1_b[1],wnum,snk_1_nq[1] ,Nw2,Nq)];
         b_snk_color_weights(1, 1, wnum, nperm, 0) = snk_color_weights_r2[index_3d(snk_2_b[1],wnum,snk_2_nq[1] ,Nw2,Nq)];
         b_snk_spin_weights(1, 1, wnum, nperm, 0) = snk_spin_weights_r2[index_3d(snk_2_b[1],wnum,snk_2_nq[1] ,Nw2,Nq)];
         b_snk_color_weights(1, 2, wnum, nperm, 0) = snk_color_weights_r2[index_3d(snk_3_b[1],wnum,snk_3_nq[1] ,Nw2,Nq)];
         b_snk_spin_weights(1, 2, wnum, nperm, 0) = snk_spin_weights_r2[index_3d(snk_3_b[1],wnum,snk_3_nq[1] ,Nw2,Nq)]; 

         b_snk_color_weights(0, 0, wnum, nperm, 1) = snk_color_weights_r1[index_3d(snk_1_b[0],wnum,snk_1_nq[0] ,Nw2,Nq)];
         b_snk_spin_weights(0, 0, wnum, nperm, 1) = snk_spin_weights_r1[index_3d(snk_1_b[0],wnum,snk_1_nq[0] ,Nw2,Nq)];
         b_snk_color_weights(0, 1, wnum, nperm, 1) = snk_color_weights_r1[index_3d(snk_2_b[0],wnum,snk_2_nq[0] ,Nw2,Nq)];
         b_snk_spin_weights(0, 1, wnum, nperm, 1) = snk_spin_weights_r1[index_3d(snk_2_b[0],wnum,snk_2_nq[0] ,Nw2,Nq)];
         b_snk_color_weights(0, 2, wnum, nperm, 1) = snk_color_weights_r1[index_3d(snk_3_b[0],wnum,snk_3_nq[0] ,Nw2,Nq)];
         b_snk_spin_weights(0, 2, wnum, nperm, 1) = snk_spin_weights_r1[index_3d(snk_3_b[0],wnum,snk_3_nq[0] ,Nw2,Nq)];
         b_snk_color_weights(1, 0, wnum, nperm, 1) = snk_color_weights_r1[index_3d(snk_1_b[1],wnum,snk_1_nq[1] ,Nw2,Nq)];
         b_snk_spin_weights(1, 0, wnum, nperm, 1) = snk_spin_weights_r1[index_3d(snk_1_b[1],wnum,snk_1_nq[1] ,Nw2,Nq)];
         b_snk_color_weights(1, 1, wnum, nperm, 1) = snk_color_weights_r1[index_3d(snk_2_b[1],wnum,snk_2_nq[1] ,Nw2,Nq)];
         b_snk_spin_weights(1, 1, wnum, nperm, 1) = snk_spin_weights_r1[index_3d(snk_2_b[1],wnum,snk_2_nq[1] ,Nw2,Nq)];
         b_snk_color_weights(1, 2, wnum, nperm, 1) = snk_color_weights_r1[index_3d(snk_3_b[1],wnum,snk_3_nq[1] ,Nw2,Nq)];
         b_snk_spin_weights(1, 2, wnum, nperm, 1) = snk_spin_weights_r1[index_3d(snk_3_b[1],wnum,snk_3_nq[1] ,Nw2,Nq)]; 

         b_snk_color_weights(0, 0, wnum, nperm, 2) = snk_color_weights_r2[index_3d(snk_1_b[0],wnum,snk_1_nq[0] ,Nw2,Nq)];
         b_snk_spin_weights(0, 0, wnum, nperm, 2) = snk_spin_weights_r2[index_3d(snk_1_b[0],wnum,snk_1_nq[0] ,Nw2,Nq)];
         b_snk_color_weights(0, 1, wnum, nperm, 2) = snk_color_weights_r2[index_3d(snk_2_b[0],wnum,snk_2_nq[0] ,Nw2,Nq)];
         b_snk_spin_weights(0, 1, wnum, nperm, 2) = snk_spin_weights_r2[index_3d(snk_2_b[0],wnum,snk_2_nq[0] ,Nw2,Nq)];
         b_snk_color_weights(0, 2, wnum, nperm, 2) = snk_color_weights_r2[index_3d(snk_3_b[0],wnum,snk_3_nq[0] ,Nw2,Nq)];
         b_snk_spin_weights(0, 2, wnum, nperm, 2) = snk_spin_weights_r2[index_3d(snk_3_b[0],wnum,snk_3_nq[0] ,Nw2,Nq)];
         b_snk_color_weights(1, 0, wnum, nperm, 2) = snk_color_weights_r2[index_3d(snk_1_b[1],wnum,snk_1_nq[1] ,Nw2,Nq)];
         b_snk_spin_weights(1, 0, wnum, nperm, 2) = snk_spin_weights_r2[index_3d(snk_1_b[1],wnum,snk_1_nq[1] ,Nw2,Nq)];
         b_snk_color_weights(1, 1, wnum, nperm, 2) = snk_color_weights_r2[index_3d(snk_2_b[1],wnum,snk_2_nq[1] ,Nw2,Nq)];
         b_snk_spin_weights(1, 1, wnum, nperm, 2) = snk_spin_weights_r2[index_3d(snk_2_b[1],wnum,snk_2_nq[1] ,Nw2,Nq)];
         b_snk_color_weights(1, 2, wnum, nperm, 2) = snk_color_weights_r2[index_3d(snk_3_b[1],wnum,snk_3_nq[1] ,Nw2,Nq)];
         b_snk_spin_weights(1, 2, wnum, nperm, 2) = snk_spin_weights_r2[index_3d(snk_3_b[1],wnum,snk_3_nq[1] ,Nw2,Nq)];

         b_snk_color_weights(0, 0, wnum, nperm, 3) = snk_color_weights_r3[index_3d(snk_1_b[0],wnum,snk_1_nq[0] ,Nw2,Nq)];
         b_snk_spin_weights(0, 0, wnum, nperm, 3) = snk_spin_weights_r3[index_3d(snk_1_b[0],wnum,snk_1_nq[0] ,Nw2,Nq)];
         b_snk_color_weights(0, 1, wnum, nperm, 3) = snk_color_weights_r3[index_3d(snk_2_b[0],wnum,snk_2_nq[0] ,Nw2,Nq)];
         b_snk_spin_weights(0, 1, wnum, nperm, 3) = snk_spin_weights_r3[index_3d(snk_2_b[0],wnum,snk_2_nq[0] ,Nw2,Nq)];
         b_snk_color_weights(0, 2, wnum, nperm, 3) = snk_color_weights_r3[index_3d(snk_3_b[0],wnum,snk_3_nq[0] ,Nw2,Nq)];
         b_snk_spin_weights(0, 2, wnum, nperm, 3) = snk_spin_weights_r3[index_3d(snk_3_b[0],wnum,snk_3_nq[0] ,Nw2,Nq)];
         b_snk_color_weights(1, 0, wnum, nperm, 3) = snk_color_weights_r3[index_3d(snk_1_b[1],wnum,snk_1_nq[1] ,Nw2,Nq)];
         b_snk_spin_weights(1, 0, wnum, nperm, 3) = snk_spin_weights_r3[index_3d(snk_1_b[1],wnum,snk_1_nq[1] ,Nw2,Nq)];
         b_snk_color_weights(1, 1, wnum, nperm, 3) = snk_color_weights_r3[index_3d(snk_2_b[1],wnum,snk_2_nq[1] ,Nw2,Nq)];
         b_snk_spin_weights(1, 1, wnum, nperm, 3) = snk_spin_weights_r3[index_3d(snk_2_b[1],wnum,snk_2_nq[1] ,Nw2,Nq)];
         b_snk_color_weights(1, 2, wnum, nperm, 3) = snk_color_weights_r3[index_3d(snk_3_b[1],wnum,snk_3_nq[1] ,Nw2,Nq)];
         b_snk_spin_weights(1, 2, wnum, nperm, 3) = snk_spin_weights_r3[index_3d(snk_3_b[1],wnum,snk_3_nq[1] ,Nw2,Nq)];
      }
      for (int wnum=0; wnum< Nw2Hex; wnum++) {
         for (int q=0; q < 2*Nq; q++) {
            b_hex_snk_color_weights((q-(q%Nq))/Nq, q%Nq, wnum, nperm, 0) = hex_snk_color_weights_A1[index_2d(wnum,perms[index_2d(nperm,q ,2*Nq)]-1 ,2*Nq)];
            b_hex_snk_spin_weights((q-(q%Nq))/Nq, q%Nq, wnum, nperm, 0) = hex_snk_spin_weights_A1[index_2d(wnum,perms[index_2d(nperm,q ,2*Nq)]-1 ,2*Nq)];
            b_hex_snk_color_weights((q-(q%Nq))/Nq, q%Nq, wnum, nperm, 1) = hex_snk_color_weights_T1_r1[index_2d(wnum,perms[index_2d(nperm,q ,2*Nq)]-1 ,2*Nq)];
            b_hex_snk_spin_weights((q-(q%Nq))/Nq, q%Nq, wnum, nperm, 1) = hex_snk_spin_weights_T1_r1[index_2d(wnum,perms[index_2d(nperm,q ,2*Nq)]-1 ,2*Nq)];
            b_hex_snk_color_weights((q-(q%Nq))/Nq, q%Nq, wnum, nperm, 2) = hex_snk_color_weights_T1_r2[index_2d(wnum,perms[index_2d(nperm,q ,2*Nq)]-1 ,2*Nq)];
            b_hex_snk_spin_weights((q-(q%Nq))/Nq, q%Nq, wnum, nperm, 2) = hex_snk_spin_weights_T1_r2[index_2d(wnum,perms[index_2d(nperm,q ,2*Nq)]-1 ,2*Nq)];
            b_hex_snk_color_weights((q-(q%Nq))/Nq, q%Nq, wnum, nperm, 3) = hex_snk_color_weights_T1_r3[index_2d(wnum,perms[index_2d(nperm,q ,2*Nq)]-1 ,2*Nq)];
            b_hex_snk_spin_weights((q-(q%Nq))/Nq, q%Nq, wnum, nperm, 3) = hex_snk_spin_weights_T1_r3[index_2d(wnum,perms[index_2d(nperm,q ,2*Nq)]-1 ,2*Nq)];
         }
      }
   }
   for (int wnum=0; wnum< Nw2Hex; wnum++) {
      b_hex_snk_weights(wnum, 0) = hex_snk_weights_A1[wnum];
      b_hex_snk_weights(wnum, 1) = hex_snk_weights_T1_r1[wnum];
      b_hex_snk_weights(wnum, 2) = hex_snk_weights_T1_r2[wnum];
      b_hex_snk_weights(wnum, 3) = hex_snk_weights_T1_r3[wnum];
   }

   for (int rp=0; rp<B2Nrows; rp++)
      for (int m=0; m<Nsrc+NsrcHex; m++)
         for (int r=0; r<B2Nrows; r++)
            for (int n=0; n<Nsnk+NsnkHex; n++)
               for (int t=0; t<Lt; t++) 
                  for (int x=0; x<Vsnk/sites_per_rank; x++) {
                     b_C_r(n,r,m,rp,x,t) = 0.0;
                     b_C_i(n,r,m,rp,x,t) = 0.0;
                 } 

   if (rank == 0) {
   printf("prop 1 %4.9f + I %4.9f \n", b_B1_prop_r(0,0,0,0,0,0,0,0), b_B1_prop_i(0,0,0,0,0,0,0,0));
   printf("prop 2 %4.9f + I %4.9f \n", b_B2_prop_r(0,0,0,0,0,0,0,0), b_B2_prop_i(0,0,0,0,0,0,0,0));
   printf("psi src 1 %4.9f + I %4.9f \n", b_B1_src_psi_r(0,0), b_B1_src_psi_i(0,0));
   printf("psi src 2 %4.9f + I %4.9f \n", b_B2_src_psi_r(0,0), b_B2_src_psi_i(0,0));
   printf("psi snk %4.9f + I %4.9f \n", b_snk_psi_r(0,0,0), b_snk_psi_i(0,0,0));
   printf("weights snk %4.1f \n", b_snk_weights(0,0));
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
				    b_src_spins.raw_buffer(),
				    b_src_spin_block_weights.raw_buffer(),
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

   printf("done \n");


   if (rank == 0) {
   printf("BB-BB non-zero? %4.9e + I %4.9e \n", b_C_r(0,0,0,0,0,0), b_C_i(0,0,0,0,0,0) );
   printf("BB-BB non-zero? %4.9e + I %4.9e \n", b_C_r(0,1,0,0,0,0), b_C_i(0,1,0,0,0,0) );
   printf("BB-BB non-zero? %4.9e + I %4.9e \n", b_C_r(0,2,0,0,0,0), b_C_i(0,2,0,0,0,0) );
   printf("BB-BB non-zero? %4.9e + I %4.9e \n", b_C_r(0,3,0,0,0,0), b_C_i(0,3,0,0,0,0) );
   printf("BB-BB non-zero? %4.9e + I %4.9e \n", b_C_r(0,4,0,0,0,0), b_C_i(0,4,0,0,0,0) );
   printf("BB-BB non-zero? %4.9e + I %4.9e \n", b_C_r(0,5,0,0,0,0), b_C_i(0,5,0,0,0,0) );
   printf("H-BB non-zero? %4.9e + I %4.9e \n", b_C_r(0,0,Nsrc,0,0,0), b_C_i(0,0,Nsrc,0,0,0) );
   printf("H-BB non-zero? %4.9e + I %4.9e \n", b_C_r(0,1,Nsrc,0,0,0), b_C_i(0,1,Nsrc,0,0,0) );
   printf("H-BB non-zero? %4.9e + I %4.9e \n", b_C_r(0,2,Nsrc,0,0,0), b_C_i(0,2,Nsrc,0,0,0) );
   printf("H-BB non-zero? %4.9e + I %4.9e \n", b_C_r(0,3,Nsrc,0,0,0), b_C_i(0,3,Nsrc,0,0,0) );
   printf("H-BB non-zero? %4.9e + I %4.9e \n", b_C_r(0,4,Nsrc,0,0,0), b_C_i(0,4,Nsrc,0,0,0) );
   printf("H-BB non-zero? %4.9e + I %4.9e \n", b_C_r(0,5,Nsrc,0,0,0), b_C_i(0,5,Nsrc,0,0,0) );
   printf("H-H non-zero? %4.9e + I %4.9e \n", b_C_r(Nsnk,0,Nsrc,0,0,0), b_C_i(Nsnk,0,Nsrc,0,0,0) );
   printf("H-H non-zero? %4.9e + I %4.9e \n", b_C_r(Nsnk,1,Nsrc,0,0,0), b_C_i(Nsnk,1,Nsrc,0,0,0) );
   printf("H-H non-zero? %4.9e + I %4.9e \n", b_C_r(Nsnk,2,Nsrc,0,0,0), b_C_i(Nsnk,2,Nsrc,0,0,0) );
   printf("H-H non-zero? %4.9e + I %4.9e \n", b_C_r(Nsnk,3,Nsrc,0,0,0), b_C_i(Nsnk,3,Nsrc,0,0,0) );
   printf("H-H non-zero? %4.9e + I %4.9e \n", b_C_r(Nsnk,4,Nsrc,0,0,0), b_C_i(Nsnk,4,Nsrc,0,0,0) );
   printf("H-H non-zero? %4.9e + I %4.9e \n", b_C_r(Nsnk,5,Nsrc,0,0,0), b_C_i(Nsnk,5,Nsrc,0,0,0) ); 
   }

    // symmetrize and such
#ifdef WITH_MPI
   for (int rp=0; rp<B2Nrows; rp++)
      for (int m=0; m<Nsrc+NsrcHex; m++)
         for (int r=0; r<B2Nrows; r++)
            for (int n=0; n<Nsnk+NsnkHex; n++)
               for (int t=0; t<Lt; t++) {
                  double number0r;
                  double number0i;
                  double this_number0r = b_C_r(n,r,m,rp,rank,t);
                  double this_number0i = b_C_i(n,r,m,rp,rank,t);
                  MPI_Allreduce(&this_number0r, &number0r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                  MPI_Allreduce(&this_number0i, &number0i, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                  C_re[index_5d(rp,m,r,n,t, Nsrc+NsrcHex,B2Nrows,Nsnk+NsnkHex,Lt)] += number0r;
                  C_im[index_5d(rp,m,r,n,t, Nsrc+NsrcHex,B2Nrows,Nsnk+NsnkHex,Lt)] += number0i;
            }
#else
   for (int rp=0; rp<B2Nrows; rp++)
      for (int m=0; m<Nsrc+NsrcHex; m++)
         for (int r=0; r<B2Nrows; r++)
            for (int n=0; n<Nsnk+NsnkHex; n++)
               for (int t=0; t<Lt; t++)
                  for (int x=0; x<Vsnk; x++) {
                     double number0r = b_C_r(n,r,m,rp,x,t);
                     double number0i = b_C_i(n,r,m,rp,x,t);
                     C_re[index_5d(rp,m,r,n,t, Nsrc+NsrcHex,B2Nrows,Nsnk+NsnkHex,Lt)] += number0r;
                     C_im[index_5d(rp,m,r,n,t, Nsrc+NsrcHex,B2Nrows,Nsnk+NsnkHex,Lt)] += number0i;
                  }
#endif

   if (rank == 0) {
   for (int b=0; b<B2Nrows; b++) {
      printf("\n");
      for (int m=0; m<Nsrc+NsrcHex; m++)
         for (int n=0; n<Nsnk+NsnkHex; n++)
            for (int t=0; t<Lt; t++) {
                  printf("r=%d, m=%d, n=%d, t=%d: %4.1f + I (%4.1f) \n", b, m, n, t, C_re[index_5d(b,m,b,n,t, Nsrc+NsrcHex,B2Nrows,Nsnk+NsnkHex,Lt)],  C_im[index_5d(b,m,b,n,t, Nsrc+NsrcHex,B2Nrows,Nsnk+NsnkHex,Lt)]);
            }
   }
   }
}

int main(int, char **)
{
   int rank = 0;
#ifdef WITH_MPI
   rank = tiramisu_MPI_init();
#endif

   srand(0);

   std::vector<std::chrono::duration<double,std::milli>> duration_vector_1;
   std::vector<std::chrono::duration<double,std::milli>> duration_vector_2;

   int q, t, iC, iS, jC, jS, y, x, x1, x2, m, n, k, wnum, nperm, b, r, rp;
   int iC1, iS1, iC2, iS2, jC1, jS1, jC2, jS2, kC1, kS1, kC2, kS2;

   // Initialization
   // Props
   double* B1_prop_re = (double *) malloc(Nq * Lt * Nc * Ns * Nc * Ns * Vsnk * Vsrc * sizeof (double));
   double* B1_prop_im = (double *) malloc(Nq * Lt * Nc * Ns * Nc * Ns * Vsnk * Vsrc * sizeof (double));
   double* B2_prop_re = (double *) malloc(Nq * Lt * Nc * Ns * Nc * Ns * Vsnk * Vsrc * sizeof (double));
   double* B2_prop_im = (double *) malloc(Nq * Lt * Nc * Ns * Nc * Ns * Vsnk * Vsrc * sizeof (double));
   for (q = 0; q < Nq; q++) {
      for (t = 0; t < Lt; t++) {
         for (iC = 0; iC < Nc; iC++) {
            for (iS = 0; iS < Ns; iS++) {
               for (jC = 0; jC < Nc; jC++) {
                  for (jS = 0; jS < Ns; jS++) {
                     for (y = 0; y < Vsrc; y++) {
                        for (x = 0; x < Vsnk; x++) {
			   if (randommode == 1) {
	                        double v1 = rand()%10;
	                        double v2 = rand()%10;
	                        double v3 = rand()%10;
	                        double v4 = rand()%10;
                           B1_prop_re[prop_index(q,t,jC,jS,iC,iS,y,x ,Nc,Ns,Vsrc,Vsnk,Lt)] = v1;
                           B2_prop_re[prop_index(q,t,jC,jS,iC,iS,y,x ,Nc,Ns,Vsrc,Vsnk,Lt)] = v2;
                           B1_prop_im[prop_index(q,t,jC,jS,iC,iS,y,x ,Nc,Ns,Vsrc,Vsnk,Lt)] = v3;
                           B2_prop_im[prop_index(q,t,jC,jS,iC,iS,y,x ,Nc,Ns,Vsrc,Vsnk,Lt)] = v4;
			   }
			   else {
                           if ((jC == iC) && (jS == iS)) {
                              B1_prop_re[prop_index(q,t,jC,jS,iC,iS,y,x ,Nc,Ns,Vsrc,Vsnk,Lt)] = 1/mq*cos(2*M_PI/6);
                              B2_prop_re[prop_index(q,t,jC,jS,iC,iS,y,x ,Nc,Ns,Vsrc,Vsnk,Lt)] = 1/mq*cos(2*M_PI/6);
                              B1_prop_im[prop_index(q,t,jC,jS,iC,iS,y,x ,Nc,Ns,Vsrc,Vsnk,Lt)] = 1/mq*sin(2*M_PI/6);
                              B2_prop_im[prop_index(q,t,jC,jS,iC,iS,y,x ,Nc,Ns,Vsrc,Vsnk,Lt)] = 1/mq*sin(2*M_PI/6); 
                           }
                           else {
                              B1_prop_re[prop_index(q,t,jC,jS,iC,iS,y,x ,Nc,Ns,Vsrc,Vsnk,Lt)] = 0;
                              B2_prop_re[prop_index(q,t,jC,jS,iC,iS,y,x ,Nc,Ns,Vsrc,Vsnk,Lt)] = 0;
                              B1_prop_im[prop_index(q,t,jC,jS,iC,iS,y,x ,Nc,Ns,Vsrc,Vsnk,Lt)] = 0;
                              B2_prop_im[prop_index(q,t,jC,jS,iC,iS,y,x ,Nc,Ns,Vsrc,Vsnk,Lt)] = 0;
                           }
			   }
                        }
                     }
                  }
               }
            }
         }
      }
   }
   // Wavefunctions
   double* src_psi_B1_re = (double *) malloc(Nsrc * Vsrc * sizeof (double));
   double* src_psi_B1_im = (double *) malloc(Nsrc * Vsrc * sizeof (double));
   double* src_psi_B2_re = (double *) malloc(Nsrc * Vsrc * sizeof (double));
   double* src_psi_B2_im = (double *) malloc(Nsrc * Vsrc * sizeof (double));
   for (m = 0; m < Nsrc; m++)
      for (x = 0; x < Vsrc; x++) {
	      double v1 = 1.0;
	      double v2 = 0.0;
	      double v3 = 1.0;
	      double v4 = 0.0;
	      if (randommode == 1) {
	      v1 = rand()%10;
	      v2 = rand()%10;
	      v3 = rand()%10;
	      v4 = rand()%10;
	      }
         src_psi_B1_re[index_2d(x,m ,Nsrc)] = v1 ;// / Vsrc;
         src_psi_B1_im[index_2d(x,m ,Nsrc)] = v2 ;// / Vsrc;
         src_psi_B2_re[index_2d(x,m ,Nsrc)] = v3 ;// / Vsrc;
         src_psi_B2_im[index_2d(x,m ,Nsrc)] = v4 ;// / Vsrc;
      }
   double* snk_psi_re = (double *) malloc(Vsnk * Vsnk * NEntangled * sizeof (double));
   double* snk_psi_im = (double *) malloc(Vsnk * Vsnk * NEntangled * sizeof (double));
   double* all_snk_psi_re = (double *) malloc(Vsnk * Vsnk * Nsnk * sizeof (double));
   double* all_snk_psi_im = (double *) malloc(Vsnk * Vsnk * Nsnk * sizeof (double));
   double* snk_psi_B1_re = (double *) malloc(Nsnk * Vsnk * sizeof (double));
   double* snk_psi_B1_im = (double *) malloc(Nsnk * Vsnk * sizeof (double));
   double* snk_psi_B2_re = (double *) malloc(Nsnk * Vsnk * sizeof (double));
   double* snk_psi_B2_im = (double *) malloc(Nsnk * Vsnk * sizeof (double));
   for (n = 0; n < Nsnk; n++) {
      for (x = 0; x < Vsnk; x++) {
	      double v1 = 1.0;
	      double v2 = 0.0;
	      double v3 = 1.0;
	      double v4 = 0.0;
	      if (randommode == 1) {
	      v1 = rand()%10;
	      v2 = rand()%10;
	      v3 = rand()%10;
	      v4 = rand()%10;
	      }
         snk_psi_B1_re[index_2d(x,n ,Nsnk)] = v1  ;// / Vsnk;
         snk_psi_B1_im[index_2d(x,n ,Nsnk)] = v2 ;// / Vsnk;
         snk_psi_B2_re[index_2d(x,n ,Nsnk)] = v3 ;// / Vsnk;
         snk_psi_B2_im[index_2d(x,n ,Nsnk)] = v4 ;// / Vsnk;
      }
   }
   for (n = 0; n < NEntangled; n++)
      for (x1 = 0; x1 < Vsnk; x1++)
         for (x2 = 0; x2 < Vsnk; x2++) {
            snk_psi_re[index_3d(x1,x2,n ,Vsnk,NEntangled)] = snk_psi_B1_re[index_2d(x1,n ,Nsnk)]*snk_psi_B2_re[index_2d(x2,n ,Nsnk)] - snk_psi_B1_im[index_2d(x1,n ,Nsnk)]*snk_psi_B2_im[index_2d(x2,n ,Nsnk)] + snk_psi_B1_re[index_2d(x2,n ,Nsnk)]*snk_psi_B2_re[index_2d(x1,n ,Nsnk)] - snk_psi_B1_im[index_2d(x2,n ,Nsnk)]*snk_psi_B2_im[index_2d(x1,n ,Nsnk)];// / Vsnk;
            snk_psi_im[index_3d(x1,x2,n ,Vsnk,NEntangled)] = snk_psi_B1_re[index_2d(x1,n ,Nsnk)]*snk_psi_B2_im[index_2d(x2,n ,Nsnk)] + snk_psi_B1_im[index_2d(x1,n ,Nsnk)]*snk_psi_B2_re[index_2d(x2,n ,Nsnk)] + snk_psi_B1_re[index_2d(x2,n ,Nsnk)]*snk_psi_B2_im[index_2d(x1,n ,Nsnk)] + snk_psi_B1_im[index_2d(x2,n ,Nsnk)]*snk_psi_B2_re[index_2d(x1,n ,Nsnk)];// / Vsnk;
            //snk_psi_re[index_3d(x1,x2,n ,Vsnk,Nsnk)] = 1;// / Vsnk;
            //snk_psi_im[index_3d(x1,x2,n ,Vsnk,Nsnk)] = 0;// / Vsnk;
         } 
   for (n = 0; n < Nsnk; n++)
      for (x1 = 0; x1 < Vsnk; x1++)
         for (x2 = 0; x2 < Vsnk; x2++) {
            all_snk_psi_re[index_3d(x1,x2,n ,Vsnk,Nsnk)] = snk_psi_B1_re[index_2d(x1,n ,Nsnk)]*snk_psi_B2_re[index_2d(x2,n ,Nsnk)] - snk_psi_B1_im[index_2d(x1,n ,Nsnk)]*snk_psi_B2_im[index_2d(x2,n ,Nsnk)] + snk_psi_B1_re[index_2d(x2,n ,Nsnk)]*snk_psi_B2_re[index_2d(x1,n ,Nsnk)] - snk_psi_B1_im[index_2d(x2,n ,Nsnk)]*snk_psi_B2_im[index_2d(x1,n ,Nsnk)];// / Vsnk;
            all_snk_psi_im[index_3d(x1,x2,n ,Vsnk,Nsnk)] = snk_psi_B1_re[index_2d(x1,n ,Nsnk)]*snk_psi_B2_im[index_2d(x2,n ,Nsnk)] + snk_psi_B1_im[index_2d(x1,n ,Nsnk)]*snk_psi_B2_re[index_2d(x2,n ,Nsnk)] + snk_psi_B1_re[index_2d(x2,n ,Nsnk)]*snk_psi_B2_im[index_2d(x1,n ,Nsnk)] + snk_psi_B1_im[index_2d(x2,n ,Nsnk)]*snk_psi_B2_re[index_2d(x1,n ,Nsnk)];// / Vsnk;
            //snk_psi_re[index_3d(x1,x2,n ,Vsnk,Nsnk)] = 1;// / Vsnk;
            //snk_psi_im[index_3d(x1,x2,n ,Vsnk,Nsnk)] = 0;// / Vsnk;
         } 
   double* hex_src_psi_re = (double *) malloc(NsrcHex * Vsrc * sizeof (double));
   double* hex_src_psi_im = (double *) malloc(NsrcHex * Vsrc * sizeof (double));
   double* hex_snk_psi_re = (double *) malloc(NsnkHex * Vsnk * sizeof (double));
   double* hex_snk_psi_im = (double *) malloc(NsnkHex * Vsnk * sizeof (double));
   for (k = 0; k < NsrcHex; k++) {
      for (y = 0; y < Vsrc; y++) {
	      double v1 = 1.0;
	      double v2 = 0.0;
	      if (randommode == 1) {
	      v1 = rand()%10;
	      v2 = rand()%10;
	      }
         hex_src_psi_re[index_2d(y,k ,NsrcHex)] = v1 ;// / Vsrc;
         hex_src_psi_im[index_2d(y,k ,NsrcHex)] = v2 ;// / Vsrc;
      }
   }
   for (k = 0; k < NsnkHex; k++) {
      for (x = 0; x < Vsnk; x++) {
	      double v1 = 1.0;
	      double v2 = 0.0;
	      if (randommode == 1) {
	      v1 = rand()%10;
	      v2 = rand()%10;
	      }
         hex_snk_psi_re[index_2d(x,k ,NsnkHex)] = v1 ;// / Vsnk;
         hex_snk_psi_im[index_2d(x,k ,NsnkHex)] = v2 ;// / Vsnk;
      }
   }
   // Weights
   /*static int src_color_weights_r1_P[Nw][Nq] = { {0,1,2}, {0,2,1}, {1,0,2} ,{0,1,2}, {0,2,1}, {1,0,2}, {1,2,0}, {2,1,0}, {2,0,1} };
   static int src_spin_weights_r1_P[Nw][Nq] = { {0,1,0}, {0,1,0}, {0,1,0}, {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0} };
   static double src_weights_r1_P[Nw] = {-2/ sqrt(2), 2/sqrt(2), 2/sqrt(2), 1/sqrt(2), -1/sqrt(2), -1/sqrt(2), 1/sqrt(2), -1/sqrt(2), 1/sqrt(2)};
   
   static int src_color_weights_r2_P[Nw][Nq] = { {0,1,2}, {0,2,1}, {1,0,2} ,{1,2,0}, {2,1,0}, {2,0,1}, {0,1,2}, {0,2,1}, {1,0,2} };
   static int src_spin_weights_r2_P[Nw][Nq] = { {0,1,1}, {0,1,1}, {0,1,1}, {0,1,1}, {0,1,1}, {0,1,1}, {1,0,1}, {1,0,1}, {1,0,1} };
   static double src_weights_r2_P[Nw] = {1/ sqrt(2), -1/sqrt(2), -1/sqrt(2), 1/sqrt(2), -1/sqrt(2), 1/sqrt(2), -2/sqrt(2), 2/sqrt(2), 2/sqrt(2)}; */
   
   static int src_color_weights_r1_P[Nw][Nq] = { {0,1,2}, {0,2,1}, {1,0,2}, {1,2,0}, {2,1,0}, {2,0,1}, {0,1,2}, {0,2,1}, {1,0,2}, {1,2,0}, {2,1,0}, {2,0,1} };
   static int src_spin_weights_r1_P[Nw][Nq] = { {0,1,0}, {0,1,0}, {0,1,0}, {0,1,0}, {0,1,0}, {0,1,0}, {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0} };
   static double src_weights_r1_P[Nw] = {-1/ sqrt(2), 1/sqrt(2), 1/sqrt(2), -1/sqrt(2), 1/sqrt(2), -1/sqrt(2), 1/ sqrt(2), -1/sqrt(2), -1/sqrt(2), 1/sqrt(2), -1/sqrt(2), 1/sqrt(2)};
    
   static int src_color_weights_r2_P[Nw][Nq] = { {0,1,2}, {0,2,1}, {1,0,2}, {1,2,0}, {2,1,0}, {2,0,1}, {0,1,2}, {0,2,1}, {1,0,2}, {1,2,0}, {2,1,0}, {2,0,1} };
   static int src_spin_weights_r2_P[Nw][Nq] = { {0,1,1}, {0,1,1}, {0,1,1}, {0,1,1}, {0,1,1}, {0,1,1}, {1,0,1}, {1,0,1}, {1,0,1}, {1,0,1}, {1,0,1}, {1,0,1} };
   static double src_weights_r2_P[Nw] = {1/ sqrt(2), -1/sqrt(2), -1/sqrt(2), 1/sqrt(2), -1/sqrt(2), 1/sqrt(2), -1/ sqrt(2), 1/sqrt(2), 1/sqrt(2), -1/sqrt(2), 1/sqrt(2), -1/sqrt(2)};

   int* src_color_weights_r1 = (int *) malloc(Nw * Nq * sizeof (int));
   int* src_color_weights_r2 = (int *) malloc(Nw * Nq * sizeof (int));
   int* src_spin_weights_r1 = (int *) malloc(Nw * Nq * sizeof (int));
   int* src_spin_weights_r2 = (int *) malloc(Nw * Nq * sizeof (int));
   double src_weights_r1[Nw];
   double src_weights_r2[Nw];
   for (wnum = 0; wnum < Nw; wnum++) {
      for (q = 0; q < Nq; q++) {
         src_color_weights_r1[index_2d(wnum,q ,Nq)] = src_color_weights_r1_P[wnum][q];
         src_color_weights_r2[index_2d(wnum,q ,Nq)] = src_color_weights_r2_P[wnum][q];
         src_spin_weights_r1[index_2d(wnum,q ,Nq)] = src_spin_weights_r1_P[wnum][q];
         src_spin_weights_r2[index_2d(wnum,q ,Nq)] = src_spin_weights_r2_P[wnum][q];
      }
      src_weights_r1[wnum] = src_weights_r1_P[wnum];
      src_weights_r2[wnum] = src_weights_r2_P[wnum];
   }
   int* snk_color_weights_A1 = (int *) malloc(Nw2Hex * 2*Nq * sizeof (int));
   int* snk_color_weights_T1_r1 = (int *) malloc(Nw2Hex * 2*Nq * sizeof (int));
   int* snk_color_weights_T1_r2 = (int *) malloc(Nw2Hex * 2*Nq * sizeof (int));
   int* snk_color_weights_T1_r3 = (int *) malloc(Nw2Hex * 2*Nq * sizeof (int));
   int* snk_spin_weights_A1 = (int *) malloc(Nw2Hex * 2*Nq * sizeof (int));
   int* snk_spin_weights_T1_r1 = (int *) malloc(Nw2Hex * 2*Nq * sizeof (int));
   int* snk_spin_weights_T1_r2 = (int *) malloc(Nw2Hex * 2*Nq * sizeof (int));
   int* snk_spin_weights_T1_r3 = (int *) malloc(Nw2Hex * 2*Nq * sizeof (int));
   double snk_weights_A1[Nw2Hex];
   double snk_weights_T1_r1[Nw2Hex];
   double snk_weights_T1_r2[Nw2Hex];
   double snk_weights_T1_r3[Nw2Hex];
   for (wnum = 0; wnum < Nw2Hex; wnum++) {
      for (q = 0; q < 2*Nq; q++) {
         snk_color_weights_A1[index_2d(wnum,q ,2*Nq)] = q % Nc;
         snk_color_weights_T1_r1[index_2d(wnum,q ,2*Nq)] = q % Nc;
         snk_color_weights_T1_r2[index_2d(wnum,q ,2*Nq)] = q % Nc;
         snk_color_weights_T1_r3[index_2d(wnum,q ,2*Nq)] = q % Nc;
      }
      snk_spin_weights_A1[index_2d(wnum,0 ,2*Nq)] = 0;
      snk_spin_weights_T1_r1[index_2d(wnum,0 ,2*Nq)] = 0;
      snk_spin_weights_T1_r2[index_2d(wnum,0 ,2*Nq)] = 0;
      snk_spin_weights_T1_r3[index_2d(wnum,0 ,2*Nq)] = 1;
      snk_spin_weights_A1[index_2d(wnum,1 ,2*Nq)] = 1;
      snk_spin_weights_T1_r1[index_2d(wnum,1 ,2*Nq)] = 1;
      snk_spin_weights_T1_r2[index_2d(wnum,1 ,2*Nq)] = 1;
      snk_spin_weights_T1_r3[index_2d(wnum,1 ,2*Nq)] = 0;
      snk_spin_weights_A1[index_2d(wnum,2 ,2*Nq)] = 0;
      snk_spin_weights_T1_r1[index_2d(wnum,2 ,2*Nq)] = 0;
      snk_spin_weights_T1_r2[index_2d(wnum,2 ,2*Nq)] = 0;
      snk_spin_weights_T1_r3[index_2d(wnum,2 ,2*Nq)] = 1;
      snk_spin_weights_A1[index_2d(wnum,3 ,2*Nq)] = 1;
      snk_spin_weights_T1_r1[index_2d(wnum,3 ,2*Nq)] = 0;
      snk_spin_weights_T1_r2[index_2d(wnum,3 ,2*Nq)] = 1;
      snk_spin_weights_T1_r3[index_2d(wnum,3 ,2*Nq)] = 1;
      snk_spin_weights_A1[index_2d(wnum,4 ,2*Nq)] = 0;
      snk_spin_weights_T1_r1[index_2d(wnum,4 ,2*Nq)] = 1;
      snk_spin_weights_T1_r2[index_2d(wnum,4 ,2*Nq)] = 0;
      snk_spin_weights_T1_r3[index_2d(wnum,4 ,2*Nq)] = 0;
      snk_spin_weights_A1[index_2d(wnum,5 ,2*Nq)] = 1;
      snk_spin_weights_T1_r1[index_2d(wnum,5 ,2*Nq)] = 0;
      snk_spin_weights_T1_r2[index_2d(wnum,5 ,2*Nq)] = 1;
      snk_spin_weights_T1_r3[index_2d(wnum,5 ,2*Nq)] = 1;
      snk_weights_A1[wnum] = -1.0/sqrt(2);
      snk_weights_T1_r1[wnum] = 1.0;
      snk_weights_T1_r2[wnum] = -1.0/sqrt(2);
      snk_weights_T1_r3[wnum] = 1.0;
   }
   // Permutations
   int perms_array[36][6] = { {1,2,3,4,5,6}, {1, 4, 3, 2, 5, 6}, {1, 6, 3, 2, 5, 4}, {1, 2, 3, 6, 5, 4}, {1, 4, 3, 6, 5, 2}, {1, 6, 3, 4, 5, 2}, {3, 2, 1, 4, 5, 6}, {3, 4, 1, 2, 5, 6}, {3, 6, 1, 2, 5, 4}, {3, 2, 1, 6, 5, 4}, {3, 4, 1, 6, 5, 2}, {3, 6, 1, 4, 5, 2}, {5, 2, 1, 4, 3, 6}, {5, 4, 1, 2, 3, 6}, {5, 6, 1, 2, 3, 4}, {5, 2, 1, 6, 3, 4}, {5, 4, 1, 6, 3, 2}, {5, 6, 1, 4, 3, 2}, {1, 2, 5, 4, 3, 6}, {1, 4, 5, 2, 3, 6}, {1, 6, 5, 2, 3, 4}, {1, 2, 5, 6, 3, 4}, {1, 4, 5, 6, 3, 2}, {1, 6, 5, 4, 3, 2}, {3, 2, 5, 4, 1, 6}, {3, 4, 5, 2, 1, 6}, {3, 6, 5, 2, 1, 4}, {3, 2, 5, 6, 1, 4}, {3, 4, 5, 6, 1, 2}, {3, 6, 5, 4, 1, 2}, {5, 2, 3, 4, 1, 6}, {5, 4, 3, 2, 1, 6}, {5, 6, 3, 2, 1, 4}, {5, 2, 3, 6, 1, 4}, {5, 4, 3, 6, 1, 2}, {5, 6, 3, 4, 1, 2} };
   int sigs_array[36] = {1,-1,1,-1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,1,-1,1,-1,1};
   int* perms = (int *) malloc(Nperms * 2*Nq * sizeof (int));
   int sigs[Nperms];
   int permnum = 0;
   for (int i = 0; i < 36; i++) {

      /*if (perms_array[i][0] > perms_array[i][2]) {
         continue;
      }
      else if (perms_array[i][3] > perms_array[i][5]) {
         continue;
      } 
      else {  */
         for (int q = 0; q < 2*Nq; q++) {
            perms[index_2d(permnum,q ,2*Nq)] = perms_array[i][q];
         }
         sigs[permnum] = sigs_array[i];
         permnum += 1;
      //}
   }
   // Correlators
   double* C_re = (double *) malloc(B2Nrows * (Nsrc+NsrcHex) * (Nsnk+NsnkHex) * Lt * sizeof (double));
   double* C_im = (double *) malloc(B2Nrows * (Nsrc+NsrcHex) * (Nsnk+NsnkHex) * Lt * sizeof (double));
   //double* C_re = (double *) malloc(B2Nrows * B2Nrows * (Nsrc+NsrcHex) * (Nsnk+NsnkHex) * Lt * sizeof (double));
   //double* C_im = (double *) malloc(B2Nrows * B2Nrows * (Nsrc+NsrcHex) * (Nsnk+NsnkHex) * Lt * sizeof (double));
   double* t_C_re = (double *) malloc(B2Nrows * B2Nrows * (Nsrc+NsrcHex) * (Nsnk+NsnkHex) * Lt * sizeof (double));
   double* t_C_im = (double *) malloc(B2Nrows * B2Nrows * (Nsrc+NsrcHex) * (Nsnk+NsnkHex) * Lt * sizeof (double));
   for (rp=0; rp<B2Nrows; rp++)
      for (m=0; m<Nsrc+NsrcHex; m++)
         for (r=0; r<B2Nrows; r++)
            for (n=0; n<Nsnk+NsnkHex; n++)
               for (t=0; t<Lt; t++) {
                  C_re[index_4d(rp,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Lt)] = 0.0;
                  C_im[index_4d(rp,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Lt)] = 0.0;
                  //C_re[index_5d(rp,m,r,n,t, Nsrc+NsrcHex,B2Nrows,Nsnk+NsnkHex,Lt)] = 0.0;
                  //C_im[index_5d(rp,m,r,n,t, Nsrc+NsrcHex,B2Nrows,Nsnk+NsnkHex,Lt)] = 0.0;
                  t_C_re[index_5d(rp,m,r,n,t, Nsrc+NsrcHex,B2Nrows,Nsnk+NsnkHex,Lt)] = 0.0;
                  t_C_im[index_5d(rp,m,r,n,t, Nsrc+NsrcHex,B2Nrows,Nsnk+NsnkHex,Lt)] = 0.0;
               }

   int space_symmetric = 0;
   int snk_entangled = 0;


   if (rank == 0) {
   std::cout << "Start Tiramisu code." <<  std::endl;
   }

   for (int i = 0; i < nb_tests; i++)
   {
      if (rank == 0)
         std::cout << "Run " << i << "/" << nb_tests <<  std::endl;
      auto start1 = std::chrono::high_resolution_clock::now();

       tiramisu_make_two_nucleon_2pt(t_C_re,
           t_C_im,
           B1_prop_re, 
           B1_prop_im, 
           B2_prop_re, 
           B2_prop_im, 
           src_color_weights_r1,
           src_spin_weights_r1,
           src_weights_r1,
           src_color_weights_r2,
           src_spin_weights_r2,
           src_weights_r2,
           snk_color_weights_A1,
           snk_spin_weights_A1,
           snk_weights_A1,
           snk_color_weights_T1_r1,
           snk_spin_weights_T1_r1,
           snk_weights_T1_r1,
           snk_color_weights_T1_r2,
           snk_spin_weights_T1_r2,
           snk_weights_T1_r2,
           snk_color_weights_T1_r3,
           snk_spin_weights_T1_r3,
           snk_weights_T1_r3,
           perms, 
           sigs, 
           src_psi_B1_re, 
           src_psi_B1_im, 
           src_psi_B2_re, 
           src_psi_B2_im, 
           snk_psi_re,
           snk_psi_im, 
           snk_psi_B1_re, 
           snk_psi_B1_im, 
           snk_psi_B2_re, 
           snk_psi_B2_im, 
           hex_src_psi_re, 
           hex_src_psi_im, 
           hex_snk_psi_re, 
           hex_snk_psi_im,
	   space_symmetric,
	   snk_entangled); //,
           //Nc,Ns,Vsrc,Vsnk,Lt,Nw,Nq,Nsrc,Nsnk,NsrcHex,NsnkHex,Nperms);
       
      auto end1 = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double,std::milli> duration1 = end1 - start1;
      duration_vector_1.push_back(duration1);
   }

   if (rank == 0) {
    std::cout << "End Tiramisu code." <<  std::endl;




#if RUN_REFERENCE
   std::cout << "Start reference C code." <<  std::endl;
   for (int i = 0; i < nb_tests; i++)
   {
	   std::cout << "Run " << i << "/" << nb_tests <<  std::endl;
	   auto start2 = std::chrono::high_resolution_clock::now();

      make_two_nucleon_2pt(C_re, C_im, B1_prop_re, B1_prop_im, B2_prop_re, B2_prop_im, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, snk_color_weights_A1, snk_spin_weights_A1, snk_weights_A1, snk_color_weights_T1_r1, snk_spin_weights_T1_r1, snk_weights_T1_r1, snk_color_weights_T1_r2, snk_spin_weights_T1_r2, snk_weights_T1_r2, snk_color_weights_T1_r3, snk_spin_weights_T1_r3, snk_weights_T1_r3, perms, sigs, src_psi_B1_re, src_psi_B1_im, src_psi_B2_re, src_psi_B2_im, all_snk_psi_re, all_snk_psi_im, snk_psi_B1_re, snk_psi_B1_im, snk_psi_B2_re, snk_psi_B2_im, hex_src_psi_re, hex_src_psi_im, hex_snk_psi_re, hex_snk_psi_im, space_symmetric, snk_entangled, Nc, Ns, Vsrc, Vsnk, Lt, Nw, Nw2Hex, Nq, Nsrc, Nsnk, NsrcHex, NsnkHex, Nperms);
           

      make_two_nucleon_2pt(C_re, C_im, B1_prop_re, B1_prop_im, B2_prop_re, B2_prop_im, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, snk_color_weights_A1, snk_spin_weights_A1, snk_weights_A1, snk_color_weights_T1_r1, snk_spin_weights_T1_r1, snk_weights_T1_r1, snk_color_weights_T1_r2, snk_spin_weights_T1_r2, snk_weights_T1_r2, snk_color_weights_T1_r3, snk_spin_weights_T1_r3, snk_weights_T1_r3, perms, sigs, src_psi_B2_re, src_psi_B2_im, src_psi_B1_re, src_psi_B1_im, all_snk_psi_re, all_snk_psi_im, snk_psi_B2_re, snk_psi_B2_im, snk_psi_B1_re, snk_psi_B1_im, hex_src_psi_re, hex_src_psi_im, hex_snk_psi_re, hex_snk_psi_im, space_symmetric, snk_entangled, Nc, Ns, Vsrc, Vsnk, Lt, Nw, Nw2Hex, Nq, Nsrc, Nsnk, NsrcHex, NsnkHex, Nperms);

   for (rp=0; rp<B2Nrows; rp++) {
      printf("\n");
      for (m=0; m<Nsrc+NsrcHex; m++)
    //     for (r=0; r<B2Nrows; r++)
            for (n=0; n<Nsnk+NsnkHex; n++)
               for (t=0; t<Lt; t++) {
                  printf("rp=%d, m=%d, r=%d, n=%d, t=%d: %4.1f + I (%4.1f) \n", rp, m, rp, n, t, C_re[index_4d(rp,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Lt)],  C_im[index_4d(rp,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Lt)]);
            }
   } 

	   auto end2 = std::chrono::high_resolution_clock::now();
	   std::chrono::duration<double,std::milli> duration2 = end2 - start2;
	   duration_vector_2.push_back(duration2);
   }
   std::cout << "End reference C code." <<  std::endl;
#endif

    print_time("performance_CPU.csv", "dibaryon", {"Tiramisu"}, {median(duration_vector_1)/1000.});

#if RUN_CHECK
    print_time("performance_CPU.csv", "dibaryon", {"Ref", "Tiramisu"}, {median(duration_vector_2)/1000., median(duration_vector_1)/1000.});
    std::cout << "\nSpeedup = " << median(duration_vector_2)/median(duration_vector_1) << std::endl;
    
   for (rp=0; rp<B2Nrows; rp++) {
      printf("\n");
      for (m=0; m<Nsrc+NsrcHex; m++)
         for (n=0; n<Nsnk+NsnkHex; n++)
//            for (r=0; r<B2Nrows; r++)
              for (t=0; t<Lt; t++) {
                 if ((std::abs(C_re[index_4d(rp,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Lt)] - t_C_re[index_5d(rp,m,rp,n,t, Nsrc+NsrcHex,B2Nrows,Nsnk+NsnkHex,Lt)]) >= 0.01*Vsnk*Vsnk) ||
	               (std::abs(C_im[index_4d(b,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Lt)] -  t_C_im[index_5d(rp,m,rp,n,t, Nsrc+NsrcHex,B2Nrows,Nsnk+NsnkHex,Lt)]) >= 0.01*Vsnk*Vsnk))
	            {
                  printf("rp=%d, m=%d, n=%d, t=%d: %4.1f + I (%4.1f) vs %4.1f + I (%4.1f) \n", rp, m, n, t, C_re[index_4d(rp,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Lt)], C_im[index_4d(rp,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Lt)],  t_C_re[index_5d(rp,m,rp,n,t, Nsrc+NsrcHex,B2Nrows,Nsnk+NsnkHex,Lt)],  t_C_im[index_5d(rp,m,rp,n,t, Nsrc+NsrcHex,B2Nrows,Nsnk+NsnkHex,Lt)]);
		            std::cout << "Error: different computed values for C_r or C_i!" << std::endl;
		            exit(1);
	            }
            }
   }

#endif
   printf("Finished\n");

    std::cout << "\n\n\033[1;32mSuccess: computed values are equal!\033[0m\n\n" << std::endl;
   }

#ifdef WITH_MPI
    tiramisu_MPI_cleanup();
#endif // WITH_MPI

    return 0;
}

#ifdef __cplusplus
}  // extern "C"
#endif
