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
int randommode = 1;



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
     int* hex_perms, 
     int* hex_sigs, 
     int* hex_perms_snk, 
     int* hex_sigs_snk, 
     int* hex_hex_perms, 
     int* hex_hex_sigs, 
     int* BB_pairs_src, 
     int* BB_pairs_snk, 
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
   int q, t, iC, iS, jC, jS, y, x, x1, x2, msc, nsc, m, n, k, wnum, nperm, b, r, rp;
   int iC1, iS1, iC2, iS2, jC1, jS1, jC2, jS2, kC1, kS1, kC2, kS2;

    int rank = 0;
#ifdef WITH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

   if (rank == 0) {
    long mega = 1024*1024;
    std::cout << "Array sizes" << std::endl;
    std::cout << "Prop:" <<  std::endl;
    std::cout << "	Max index size = " << Vsnk*Vsrc*Nc*Ns*Nc*Ns*Lt*Nq <<  std::endl;
    std::cout << "	Array size = " << Vsnk*Vsrc*Nc*Ns*Nc*Ns*Lt*Nq*sizeof(std::complex<double>)/mega << " Mega bytes" << std::endl;
    //std::cout << "Q, O & P:" <<  std::endl;
    //std::cout << "	Max index size = " << Vsnk*Vsrc*Nc*Ns*Nc*Ns <<  std::endl;
    //std::cout << "	Array size = " << Vsnk*Vsrc*Nc*Ns*Nc*Ns*sizeof(std::complex<double>)/mega << " Mega bytes" <<  std::endl;

    long kilo = 1024;
    //std::cout << "Blocal:" <<  std::endl;
    //std::cout << "	Max index size = " << Vsnk*Nsrc*Nc*Ns*Nc*Ns*Nc*Ns <<  std::endl;
    //std::cout << "	Array size = " << Vsnk*Nsrc*Nc*Ns*Nc*Ns*Nc*Ns*sizeof(std::complex<double>)/kilo << " kilo bytes" <<  std::endl;
    std::cout << "Blocal, Bsingle, Bdouble:" <<  std::endl;
    std::cout << "	Max index size = " << Nc*Ns*Nc*Ns*Nc*Ns <<  std::endl;
    std::cout << "	Array size = " << Nc*Ns*Nc*Ns*Nc*Ns*sizeof(std::complex<double>)/kilo << " kilo bytes" <<  std::endl;
    std::cout << std::endl;
   }

  // Halide buffers
   int NsrcTot = B1NsrcSC*Nsrc+B2NsrcSC*NsrcHex;
   int NsnkTot = B1NsnkSC*Nsnk+B2NsnkSC*NsnkHex;
   Halide::Buffer<double> b_C_r(NsnkTot, B2Nrows, NsrcTot, B2Nrows, Vsnk/sites_per_rank, Lt, "C_r");
   Halide::Buffer<double> b_C_i(NsnkTot, B2Nrows, NsrcTot, B2Nrows, Vsnk/sites_per_rank, Lt, "C_i");

   Halide::Buffer<int> b_src_color_weights(Nq, Nw, B1NsrcSC, 2, 2,  "src_color_weights");
   Halide::Buffer<int> b_src_spin_weights(Nq, Nw, B1NsrcSC, 2, 2, "src_spin_weights");
   Halide::Buffer<double> b_src_weights(Nw, B1NsrcSC, 2, 2, "src_weights");

   Halide::Buffer<int> b_src_spins(2, 2, B2Nrows, "src_spins");
   Halide::Buffer<double> b_src_spin_block_weights(2, B2Nrows, "src_spin_block_weights");
   Halide::Buffer<int> b_snk_b(2, Nq, B1NsnkSC, B1NsrcSC, Nperms, "snk_b");

   Halide::Buffer<int> b_sigs((int *)sigs, {B1NsnkSC, B1NsrcSC, Nperms});
   Halide::Buffer<int> b_hex_sigs((int *)hex_sigs, {B1NsrcSC, Nperms});
   Halide::Buffer<int> b_hex_hex_sigs((int *)hex_hex_sigs, {Nperms});

   Halide::Buffer<int> b_snk_color_weights(2, Nq, Nw2, B1NsnkSC, B1NsrcSC, Nperms, B2Nrows, "snk_color_weights");
   Halide::Buffer<int> b_snk_spin_weights(2, Nq, Nw2, B1NsnkSC, B1NsrcSC, Nperms, B2Nrows, "snk_spin_weights");
   Halide::Buffer<double> b_snk_weights(Nw2, B1NsnkSC, B2Nrows, "snk_weights");
   Halide::Buffer<int> b_hex_snk_color_weights(2, Nq, Nw2Hex, B2NsnkSC, B1NsrcSC, Nperms, B2Nrows, "hex_snk_color_weights");
   Halide::Buffer<int> b_hex_snk_spin_weights(2, Nq, Nw2Hex, B2NsnkSC, B1NsrcSC, Nperms, B2Nrows, "hex_snk_spin_weights");
   Halide::Buffer<double> b_hex_snk_weights(Nw2Hex, B2NsnkSC, B2Nrows, "hex_snk_weights");
   Halide::Buffer<int> b_hex_src_color_weights(2, Nq, Nw2Hex, B2NsrcSC, B1NsnkSC, Nperms, B2Nrows, "hex_src_color_weights");
   Halide::Buffer<int> b_hex_src_spin_weights(2, Nq, Nw2Hex, B2NsrcSC, B1NsnkSC, Nperms, B2Nrows, "hex_src_spin_weights");
   Halide::Buffer<double> b_hex_src_weights(Nw2Hex, B2NsrcSC, B2Nrows, "hex_src_weights");
   Halide::Buffer<int> b_hex_hex_snk_color_weights(2, Nq, Nw2Hex, B2NsnkSC, Nperms, B2Nrows, "hex_hex_snk_color_weights");
   Halide::Buffer<int> b_hex_hex_snk_spin_weights(2, Nq, Nw2Hex, B2NsnkSC, Nperms, B2Nrows, "hex_hex_snk_spin_weights");

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
    Halide::Buffer<double> b_snk_psi_r((double *)snk_psi_re, {NEntangled, Vsnk, sites_per_rank});
    Halide::Buffer<double> b_snk_psi_i((double *)snk_psi_im, {NEntangled, Vsnk, sites_per_rank});


   if (rank == 0) {
   printf("psi elem %4.9f \n", b_snk_psi_r(0,0,0));
   }

   // Weights
   for (int msc=0; msc<B1NsrcSC; msc++) 
      for (int wnum=0; wnum< Nw; wnum++) {
      for (int b=0; b< 2; b++) {
         int mscB = BB_pairs_src[index_2d(msc,b ,2)]-1;
         b_src_weights(wnum, msc, 0, b) = src_weights_r1[index_2d(mscB,wnum ,Nw)];
         b_src_weights(wnum, msc, 1, b) = src_weights_r2[index_2d(mscB,wnum ,Nw)];

         b_src_color_weights(0, wnum, msc, 0, b) = src_color_weights_r1[index_3d(mscB,wnum,0 ,Nw,Nq)];
         b_src_spin_weights(0, wnum, msc, 0, b) = src_spin_weights_r1[index_3d(mscB,wnum,0 ,Nw,Nq)];
         b_src_color_weights(1, wnum, msc, 0, b) = src_color_weights_r1[index_3d(mscB,wnum,1 ,Nw,Nq)];
         b_src_spin_weights(1, wnum, msc, 0, b) = src_spin_weights_r1[index_3d(mscB,wnum,1 ,Nw,Nq)];
         b_src_color_weights(2, wnum, msc, 0, b) = src_color_weights_r1[index_3d(mscB,wnum,2 ,Nw,Nq)];
         b_src_spin_weights(2, wnum, msc, 0, b) = src_spin_weights_r1[index_3d(mscB,wnum,2 ,Nw,Nq)];

         b_src_color_weights(0, wnum, msc, 1, b) = src_color_weights_r2[index_3d(mscB,wnum,0 ,Nw,Nq)];
         b_src_spin_weights(0, wnum, msc, 1, b) = src_spin_weights_r2[index_3d(mscB,wnum,0 ,Nw,Nq)];
         b_src_color_weights(1, wnum, msc, 1, b) = src_color_weights_r2[index_3d(mscB,wnum,1 ,Nw,Nq)];
         b_src_spin_weights(1, wnum, msc, 1, b) = src_spin_weights_r2[index_3d(mscB,wnum,1 ,Nw,Nq)];
         b_src_color_weights(2, wnum, msc, 1, b) = src_color_weights_r2[index_3d(mscB,wnum,2 ,Nw,Nq)];
         b_src_spin_weights(2, wnum, msc, 1, b) = src_spin_weights_r2[index_3d(mscB,wnum,2 ,Nw,Nq)];
      }
      }

   if (rank == 0 && B1NsnkSC > 0) {
   printf("src weights elem %4.9f \n", b_src_color_weights(0,0,0,0,0));
   }

   int* snk_color_weights_r1 = (int *) malloc(2 * Nw2 * B1NsnkSC * Nq * sizeof (int));
   int* snk_color_weights_r2 = (int *) malloc(2 * Nw2 * B1NsnkSC * Nq * sizeof (int));
   int* snk_color_weights_r3 = (int *) malloc(2 * Nw2 * B1NsnkSC * Nq * sizeof (int));
   int* snk_spin_weights_r1 = (int *) malloc(2 * Nw2 * B1NsnkSC * Nq * sizeof (int));
   int* snk_spin_weights_r2 = (int *) malloc(2 * Nw2 * B1NsnkSC * Nq * sizeof (int));
   int* snk_spin_weights_r3 = (int *) malloc(2 * Nw2 * B1NsnkSC * Nq * sizeof (int));
   for (int nsc=0; nsc<B1NsnkSC; nsc++) 
   for (int nB1=0; nB1<Nw; nB1++) {
      for (int nB2=0; nB2<Nw; nB2++) {
         int nscB1 = BB_pairs_snk[index_2d(nsc,0 ,2)]-1;
         int nscB2 = BB_pairs_snk[index_2d(nsc,1 ,2)]-1;
         b_snk_weights(nB1+Nw*nB2, nsc, 0) = 1.0/sqrt(2) * src_weights_r1[index_2d(nscB1,nB1 ,Nw)]*src_weights_r2[index_2d(nscB2,nB2 ,Nw)];
         b_snk_weights(nB1+Nw*nB2, nsc, 1) = src_weights_r1[index_2d(nscB1,nB1 ,Nw)]*src_weights_r1[index_2d(nscB2,nB2 ,Nw)];
         b_snk_weights(nB1+Nw*nB2, nsc, 2) = 1.0/sqrt(2) * src_weights_r1[index_2d(nscB1,nB1 ,Nw)]*src_weights_r2[index_2d(nscB2,nB2 ,Nw)];
         b_snk_weights(nB1+Nw*nB2, nsc, 3) = src_weights_r2[index_2d(nscB1,nB1 ,Nw)]*src_weights_r2[index_2d(nscB2,nB2 ,Nw)];
         for (int nq=0; nq<Nq; nq++) {
            // T1g_r1
            snk_color_weights_r1[index_4d(0,nsc,nB1+Nw*nB2,nq ,B1NsnkSC,Nw2,Nq)] = src_color_weights_r1[index_3d(nscB1,nB1,nq ,Nw,Nq)];
            snk_spin_weights_r1[index_4d(0,nsc,nB1+Nw*nB2,nq ,B1NsnkSC,Nw2,Nq)] = src_spin_weights_r1[index_3d(nscB1,nB1,nq ,Nw,Nq)];
            snk_color_weights_r1[index_4d(1,nsc,nB1+Nw*nB2,nq ,B1NsnkSC,Nw2,Nq)] = src_color_weights_r1[index_3d(nscB2,nB2,nq ,Nw,Nq)];
            snk_spin_weights_r1[index_4d(1,nsc,nB1+Nw*nB2,nq ,B1NsnkSC,Nw2,Nq)] = src_spin_weights_r1[index_3d(nscB2,nB2,nq ,Nw,Nq)];
            // T1g_r2 (and A1g)
            snk_color_weights_r2[index_4d(0,nsc,nB1+Nw*nB2,nq ,B1NsnkSC,Nw2,Nq)] = src_color_weights_r1[index_3d(nscB1,nB1,nq ,Nw,Nq)];
            snk_spin_weights_r2[index_4d(0,nsc,nB1+Nw*nB2,nq ,B1NsnkSC,Nw2,Nq)] = src_spin_weights_r1[index_3d(nscB1,nB1,nq ,Nw,Nq)];
            snk_color_weights_r2[index_4d(1,nsc,nB1+Nw*nB2,nq ,B1NsnkSC,Nw2,Nq)] = src_color_weights_r2[index_3d(nscB2,nB2,nq ,Nw,Nq)];
            snk_spin_weights_r2[index_4d(1,nsc,nB1+Nw*nB2,nq ,B1NsnkSC,Nw2,Nq)] = src_spin_weights_r2[index_3d(nscB2,nB2,nq ,Nw,Nq)];
            // T1g_r3 
            snk_color_weights_r3[index_4d(0,nsc,nB1+Nw*nB2,nq ,B1NsnkSC,Nw2,Nq)] = src_color_weights_r2[index_3d(nscB1,nB1,nq ,Nw,Nq)];
            snk_spin_weights_r3[index_4d(0,nsc,nB1+Nw*nB2,nq ,B1NsnkSC,Nw2,Nq)] = src_spin_weights_r2[index_3d(nscB1,nB1,nq ,Nw,Nq)];
            snk_color_weights_r3[index_4d(1,nsc,nB1+Nw*nB2,nq ,B1NsnkSC,Nw2,Nq)] = src_color_weights_r2[index_3d(nscB2,nB2,nq ,Nw,Nq)];
            snk_spin_weights_r3[index_4d(1,nsc,nB1+Nw*nB2,nq ,B1NsnkSC,Nw2,Nq)] = src_spin_weights_r2[index_3d(nscB2,nB2,nq ,Nw,Nq)];
         }
      }
   }
   for (int nsc=0; nsc<B1NsnkSC; nsc++) 
   for (int nB1=0; nB1<Nw; nB1++) {
      for (int nB2=0; nB2<Nw; nB2++) {
         int nscB1 = BB_pairs_snk[index_2d(nsc,0 ,2)]-1;
         int nscB2 = BB_pairs_snk[index_2d(nsc,1 ,2)]-1;
         b_snk_weights(Nw*Nw+nB1+Nw*nB2, nsc, 0) = -1.0/sqrt(2) * src_weights_r2[index_2d(nscB1,nB1 ,Nw)]*src_weights_r1[index_2d(nscB2,nB2 ,Nw)];
         b_snk_weights(Nw*Nw+nB1+Nw*nB2, nsc, 1) = 0.0;
         b_snk_weights(Nw*Nw+nB1+Nw*nB2, nsc, 2) = 1.0/sqrt(2) * src_weights_r2[index_2d(nscB1,nB1 ,Nw)]*src_weights_r1[index_2d(nscB2,nB2 ,Nw)];
         b_snk_weights(Nw*Nw+nB1+Nw*nB2, nsc, 3) = 0.0;
         for (int nq=0; nq<Nq; nq++) {
           // T1g_r1
            snk_color_weights_r1[index_4d(0,nsc,Nw*Nw+nB1+Nw*nB2,nq ,B1NsnkSC,Nw2,Nq)] = src_color_weights_r1[index_3d(nscB1,nB1,nq ,Nw,Nq)];
            snk_spin_weights_r1[index_4d(0,nsc,Nw*Nw+nB1+Nw*nB2,nq ,B1NsnkSC,Nw2,Nq)] = src_spin_weights_r1[index_3d(nscB1,nB1,nq ,Nw,Nq)];
            snk_color_weights_r1[index_4d(1,nsc,Nw*Nw+nB1+Nw*nB2,nq ,B1NsnkSC,Nw2,Nq)] = src_color_weights_r1[index_3d(nscB2,nB2,nq ,Nw,Nq)];
            snk_spin_weights_r1[index_4d(1,nsc,Nw*Nw+nB1+Nw*nB2,nq ,B1NsnkSC,Nw2,Nq)] = src_spin_weights_r1[index_3d(nscB2,nB2,nq ,Nw,Nq)];
            // T1g_r2
            snk_color_weights_r2[index_4d(0,nsc,Nw*Nw+nB1+Nw*nB2,nq ,B1NsnkSC,Nw2,Nq)] = src_color_weights_r2[index_3d(nscB1,nB1,nq ,Nw,Nq)];
            snk_spin_weights_r2[index_4d(0,nsc,Nw*Nw+nB1+Nw*nB2,nq ,B1NsnkSC,Nw2,Nq)] = src_spin_weights_r2[index_3d(nscB1,nB1,nq ,Nw,Nq)];
            snk_color_weights_r2[index_4d(1,nsc,Nw*Nw+nB1+Nw*nB2,nq ,B1NsnkSC,Nw2,Nq)] = src_color_weights_r1[index_3d(nscB2,nB2,nq ,Nw,Nq)];
            snk_spin_weights_r2[index_4d(1,nsc,Nw*Nw+nB1+Nw*nB2,nq ,B1NsnkSC,Nw2,Nq)] = src_spin_weights_r1[index_3d(nscB2,nB2,nq ,Nw,Nq)];
            // T1g_r3 
            snk_color_weights_r3[index_4d(0,nsc,Nw*Nw+nB1+Nw*nB2,nq ,B1NsnkSC,Nw2,Nq)] = src_color_weights_r2[index_3d(nscB1,nB1,nq ,Nw,Nq)];
            snk_spin_weights_r3[index_4d(0,nsc,Nw*Nw+nB1+Nw*nB2,nq ,B1NsnkSC,Nw2,Nq)] = src_spin_weights_r2[index_3d(nscB1,nB1,nq ,Nw,Nq)];
            snk_color_weights_r3[index_4d(1,nsc,Nw*Nw+nB1+Nw*nB2,nq ,B1NsnkSC,Nw2,Nq)] = src_color_weights_r2[index_3d(nscB2,nB2,nq ,Nw,Nq)];
            snk_spin_weights_r3[index_4d(1,nsc,Nw*Nw+nB1+Nw*nB2,nq ,B1NsnkSC,Nw2,Nq)] = src_spin_weights_r2[index_3d(nscB2,nB2,nq ,Nw,Nq)];
         }
      }
   }

   if (rank == 0) {
   printf("snk weights elem %4.9f \n", snk_color_weights_r1[index_4d(0,0,0,0 ,B1NsnkSC,Nw2,Nq)]);
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
      for (int msc=0; msc<B1NsrcSC; msc++)  {
      for (int nsc=0; nsc<B1NsnkSC; nsc++)  {
      for (int b=0; b<Nb; b++) {
         snk_1[b] = perms[index_4d(nperm,msc,nsc,Nq*b+0 ,B1NsrcSC,B1NsnkSC,2*Nq)] - 1;
         snk_2[b] = perms[index_4d(nperm,msc,nsc,Nq*b+1 ,B1NsrcSC,B1NsnkSC,2*Nq)] - 1;
         snk_3[b] = perms[index_4d(nperm,msc,nsc,Nq*b+2 ,B1NsrcSC,B1NsnkSC,2*Nq)] - 1;
         snk_1_b[b] = (snk_1[b] - snk_1[b] % Nq) / Nq;
         snk_2_b[b] = (snk_2[b] - snk_2[b] % Nq) / Nq;
         snk_3_b[b] = (snk_3[b] - snk_3[b] % Nq) / Nq;
         snk_1_nq[b] = snk_1[b] % Nq;
         snk_2_nq[b] = snk_2[b] % Nq;
         snk_3_nq[b] = snk_3[b] % Nq;
         b_snk_b(b, 0, nsc, msc, nperm) = snk_1_b[b];
         b_snk_b(b, 1, nsc, msc, nperm) = snk_2_b[b];
         b_snk_b(b, 2, nsc, msc, nperm) = snk_3_b[b];
      }
      for (int wnum=0; wnum< Nw2; wnum++) {
         b_snk_color_weights(0, 0, wnum, nsc, msc, nperm, 0) = snk_color_weights_r2[index_4d(snk_1_b[0],nsc,wnum,snk_1_nq[0] ,B1NsnkSC,Nw2,Nq)];
         b_snk_spin_weights(0, 0, wnum, nsc, msc, nperm, 0) = snk_spin_weights_r2[index_4d(snk_1_b[0],nsc,wnum,snk_1_nq[0] ,B1NsnkSC,Nw2,Nq)];
         b_snk_color_weights(0, 1, wnum, nsc, msc, nperm, 0) = snk_color_weights_r2[index_4d(snk_2_b[0],nsc,wnum,snk_2_nq[0] ,B1NsnkSC,Nw2,Nq)];
         b_snk_spin_weights(0, 1, wnum, nsc, msc, nperm, 0) = snk_spin_weights_r2[index_4d(snk_2_b[0],nsc,wnum,snk_2_nq[0] ,B1NsnkSC,Nw2,Nq)];
         b_snk_color_weights(0, 2, wnum, nsc, msc, nperm, 0) = snk_color_weights_r2[index_4d(snk_3_b[0],nsc,wnum,snk_3_nq[0] ,B1NsnkSC,Nw2,Nq)];
         b_snk_spin_weights(0, 2, wnum, nsc, msc, nperm, 0) = snk_spin_weights_r2[index_4d(snk_3_b[0],nsc,wnum,snk_3_nq[0] ,B1NsnkSC,Nw2,Nq)];
         b_snk_color_weights(1, 0, wnum, nsc, msc, nperm, 0) = snk_color_weights_r2[index_4d(snk_1_b[1],nsc,wnum,snk_1_nq[1] ,B1NsnkSC,Nw2,Nq)];
         b_snk_spin_weights(1, 0, wnum, nsc, msc, nperm, 0) = snk_spin_weights_r2[index_4d(snk_1_b[1],nsc,wnum,snk_1_nq[1] ,B1NsnkSC,Nw2,Nq)];
         b_snk_color_weights(1, 1, wnum, nsc, msc, nperm, 0) = snk_color_weights_r2[index_4d(snk_2_b[1],nsc,wnum,snk_2_nq[1] ,B1NsnkSC,Nw2,Nq)];
         b_snk_spin_weights(1, 1, wnum, nsc, msc, nperm, 0) = snk_spin_weights_r2[index_4d(snk_2_b[1],nsc,wnum,snk_2_nq[1] ,B1NsnkSC,Nw2,Nq)];
         b_snk_color_weights(1, 2, wnum, nsc, msc, nperm, 0) = snk_color_weights_r2[index_4d(snk_3_b[1],nsc,wnum,snk_3_nq[1] ,B1NsnkSC,Nw2,Nq)];
         b_snk_spin_weights(1, 2, wnum, nsc, msc, nperm, 0) = snk_spin_weights_r2[index_4d(snk_3_b[1],nsc,wnum,snk_3_nq[1] ,B1NsnkSC,Nw2,Nq)]; 

         b_snk_color_weights(0, 0, wnum, nsc, msc, nperm, 1) = snk_color_weights_r1[index_4d(snk_1_b[0],nsc,wnum,snk_1_nq[0] ,B1NsnkSC,Nw2,Nq)];
         b_snk_spin_weights(0, 0, wnum, nsc, msc, nperm, 1) = snk_spin_weights_r1[index_4d(snk_1_b[0],nsc,wnum,snk_1_nq[0] ,B1NsnkSC,Nw2,Nq)];
         b_snk_color_weights(0, 1, wnum, nsc, msc, nperm, 1) = snk_color_weights_r1[index_4d(snk_2_b[0],nsc,wnum,snk_2_nq[0] ,B1NsnkSC,Nw2,Nq)];
         b_snk_spin_weights(0, 1, wnum, nsc, msc, nperm, 1) = snk_spin_weights_r1[index_4d(snk_2_b[0],nsc,wnum,snk_2_nq[0] ,B1NsnkSC,Nw2,Nq)];
         b_snk_color_weights(0, 2, wnum, nsc, msc, nperm, 1) = snk_color_weights_r1[index_4d(snk_3_b[0],nsc,wnum,snk_3_nq[0] ,B1NsnkSC,Nw2,Nq)];
         b_snk_spin_weights(0, 2, wnum, nsc, msc, nperm, 1) = snk_spin_weights_r1[index_4d(snk_3_b[0],nsc,wnum,snk_3_nq[0] ,B1NsnkSC,Nw2,Nq)];
         b_snk_color_weights(1, 0, wnum, nsc, msc, nperm, 1) = snk_color_weights_r1[index_4d(snk_1_b[1],nsc,wnum,snk_1_nq[1] ,B1NsnkSC,Nw2,Nq)];
         b_snk_spin_weights(1, 0, wnum, nsc, msc, nperm, 1) = snk_spin_weights_r1[index_4d(snk_1_b[1],nsc,wnum,snk_1_nq[1] ,B1NsnkSC,Nw2,Nq)];
         b_snk_color_weights(1, 1, wnum, nsc, msc, nperm, 1) = snk_color_weights_r1[index_4d(snk_2_b[1],nsc,wnum,snk_2_nq[1] ,B1NsnkSC,Nw2,Nq)];
         b_snk_spin_weights(1, 1, wnum, nsc, msc, nperm, 1) = snk_spin_weights_r1[index_4d(snk_2_b[1],nsc,wnum,snk_2_nq[1] ,B1NsnkSC,Nw2,Nq)];
         b_snk_color_weights(1, 2, wnum, nsc, msc, nperm, 1) = snk_color_weights_r1[index_4d(snk_3_b[1],nsc,wnum,snk_3_nq[1] ,B1NsnkSC,Nw2,Nq)];
         b_snk_spin_weights(1, 2, wnum, nsc, msc, nperm, 1) = snk_spin_weights_r1[index_4d(snk_3_b[1],nsc,wnum,snk_3_nq[1] ,B1NsnkSC,Nw2,Nq)]; 

         b_snk_color_weights(0, 0, wnum, nsc, msc, nperm, 2) = snk_color_weights_r2[index_4d(snk_1_b[0],nsc,wnum,snk_1_nq[0] ,B1NsnkSC,Nw2,Nq)];
         b_snk_spin_weights(0, 0, wnum, nsc, msc, nperm, 2) = snk_spin_weights_r2[index_4d(snk_1_b[0],nsc,wnum,snk_1_nq[0] ,B1NsnkSC,Nw2,Nq)];
         b_snk_color_weights(0, 1, wnum, nsc, msc, nperm, 2) = snk_color_weights_r2[index_4d(snk_2_b[0],nsc,wnum,snk_2_nq[0] ,B1NsnkSC,Nw2,Nq)];
         b_snk_spin_weights(0, 1, wnum, nsc, msc, nperm, 2) = snk_spin_weights_r2[index_4d(snk_2_b[0],nsc,wnum,snk_2_nq[0] ,B1NsnkSC,Nw2,Nq)];
         b_snk_color_weights(0, 2, wnum, nsc, msc, nperm, 2) = snk_color_weights_r2[index_4d(snk_3_b[0],nsc,wnum,snk_3_nq[0] ,B1NsnkSC,Nw2,Nq)];
         b_snk_spin_weights(0, 2, wnum, nsc, msc, nperm, 2) = snk_spin_weights_r2[index_4d(snk_3_b[0],nsc,wnum,snk_3_nq[0] ,B1NsnkSC,Nw2,Nq)];
         b_snk_color_weights(1, 0, wnum, nsc, msc, nperm, 2) = snk_color_weights_r2[index_4d(snk_1_b[1],nsc,wnum,snk_1_nq[1] ,B1NsnkSC,Nw2,Nq)];
         b_snk_spin_weights(1, 0, wnum, nsc, msc, nperm, 2) = snk_spin_weights_r2[index_4d(snk_1_b[1],nsc,wnum,snk_1_nq[1] ,B1NsnkSC,Nw2,Nq)];
         b_snk_color_weights(1, 1, wnum, nsc, msc, nperm, 2) = snk_color_weights_r2[index_4d(snk_2_b[1],nsc,wnum,snk_2_nq[1] ,B1NsnkSC,Nw2,Nq)];
         b_snk_spin_weights(1, 1, wnum, nsc, msc, nperm, 2) = snk_spin_weights_r2[index_4d(snk_2_b[1],nsc,wnum,snk_2_nq[1] ,B1NsnkSC,Nw2,Nq)];
         b_snk_color_weights(1, 2, wnum, nsc, msc, nperm, 2) = snk_color_weights_r2[index_4d(snk_3_b[1],nsc,wnum,snk_3_nq[1] ,B1NsnkSC,Nw2,Nq)];
         b_snk_spin_weights(1, 2, wnum, nsc, msc, nperm, 2) = snk_spin_weights_r2[index_4d(snk_3_b[1],nsc,wnum,snk_3_nq[1] ,B1NsnkSC,Nw2,Nq)];

         b_snk_color_weights(0, 0, wnum, nsc, msc, nperm, 3) = snk_color_weights_r3[index_4d(snk_1_b[0],nsc,wnum,snk_1_nq[0] ,B1NsnkSC,Nw2,Nq)];
         b_snk_spin_weights(0, 0, wnum, nsc, msc, nperm, 3) = snk_spin_weights_r3[index_4d(snk_1_b[0],nsc,wnum,snk_1_nq[0] ,B1NsnkSC,Nw2,Nq)];
         b_snk_color_weights(0, 1, wnum, nsc, msc, nperm, 3) = snk_color_weights_r3[index_4d(snk_2_b[0],nsc,wnum,snk_2_nq[0] ,B1NsnkSC,Nw2,Nq)];
         b_snk_spin_weights(0, 1, wnum, nsc, msc, nperm, 3) = snk_spin_weights_r3[index_4d(snk_2_b[0],nsc,wnum,snk_2_nq[0] ,B1NsnkSC,Nw2,Nq)];
         b_snk_color_weights(0, 2, wnum, nsc, msc, nperm, 3) = snk_color_weights_r3[index_4d(snk_3_b[0],nsc,wnum,snk_3_nq[0] ,B1NsnkSC,Nw2,Nq)];
         b_snk_spin_weights(0, 2, wnum, nsc, msc, nperm, 3) = snk_spin_weights_r3[index_4d(snk_3_b[0],nsc,wnum,snk_3_nq[0] ,B1NsnkSC,Nw2,Nq)];
         b_snk_color_weights(1, 0, wnum, nsc, msc, nperm, 3) = snk_color_weights_r3[index_4d(snk_1_b[1],nsc,wnum,snk_1_nq[1] ,B1NsnkSC,Nw2,Nq)];
         b_snk_spin_weights(1, 0, wnum, nsc, msc, nperm, 3) = snk_spin_weights_r3[index_4d(snk_1_b[1],nsc,wnum,snk_1_nq[1] ,B1NsnkSC,Nw2,Nq)];
         b_snk_color_weights(1, 1, wnum, nsc, msc, nperm, 3) = snk_color_weights_r3[index_4d(snk_2_b[1],nsc,wnum,snk_2_nq[1] ,B1NsnkSC,Nw2,Nq)];
         b_snk_spin_weights(1, 1, wnum, nsc, msc, nperm, 3) = snk_spin_weights_r3[index_4d(snk_2_b[1],nsc,wnum,snk_2_nq[1] ,B1NsnkSC,Nw2,Nq)];
         b_snk_color_weights(1, 2, wnum, nsc, msc, nperm, 3) = snk_color_weights_r3[index_4d(snk_3_b[1],nsc,wnum,snk_3_nq[1] ,B1NsnkSC,Nw2,Nq)];
         b_snk_spin_weights(1, 2, wnum, nsc, msc, nperm, 3) = snk_spin_weights_r3[index_4d(snk_3_b[1],nsc,wnum,snk_3_nq[1] ,B1NsnkSC,Nw2,Nq)];
      }
      }
      }
      for (int msc=0; msc<B1NsrcSC; msc++) 
      for (int nsc=0; nsc<B2NsnkSC; nsc++) 
      for (int wnum=0; wnum< Nw2Hex; wnum++) {
         for (int q=0; q < 2*Nq; q++) {
            int b = (q-(q%Nq))/Nq;
            int mscB = msc;
            b_hex_snk_color_weights(b, q%Nq, wnum, nsc, msc, nperm, 0) = hex_snk_color_weights_A1[index_3d(nsc,wnum,hex_perms[index_3d(nperm,mscB,q ,B1NsrcSC,2*Nq)]-1 ,Nw2Hex,2*Nq)];
            b_hex_snk_spin_weights(b, q%Nq, wnum, nsc, msc, nperm, 0) = hex_snk_spin_weights_A1[index_3d(nsc,wnum,hex_perms[index_3d(nperm,mscB,q ,B1NsrcSC,2*Nq)]-1 ,Nw2Hex,2*Nq)];
            b_hex_snk_color_weights(b, q%Nq, wnum, nsc, msc, nperm, 1) = hex_snk_color_weights_T1_r1[index_3d(nsc,wnum,hex_perms[index_3d(nperm,mscB,q ,B1NsrcSC,2*Nq)]-1 ,Nw2Hex,2*Nq)];
            b_hex_snk_spin_weights(b, q%Nq, wnum, nsc, msc, nperm, 1) = hex_snk_spin_weights_T1_r1[index_3d(nsc,wnum,hex_perms[index_3d(nperm,mscB,q ,B1NsrcSC,2*Nq)]-1 ,Nw2Hex,2*Nq)];
            b_hex_snk_color_weights(b, q%Nq, wnum, nsc, msc, nperm, 2) = hex_snk_color_weights_T1_r2[index_3d(nsc,wnum,hex_perms[index_3d(nperm,mscB,q ,B1NsrcSC,2*Nq)]-1 ,Nw2Hex,2*Nq)];
            b_hex_snk_spin_weights(b, q%Nq, wnum, nsc, msc, nperm, 2) = hex_snk_spin_weights_T1_r2[index_3d(nsc,wnum,hex_perms[index_3d(nperm,mscB,q ,B1NsrcSC,2*Nq)]-1 ,Nw2Hex,2*Nq)];
            b_hex_snk_color_weights(b, q%Nq, wnum, nsc, msc, nperm, 3) = hex_snk_color_weights_T1_r3[index_3d(nsc,wnum,hex_perms[index_3d(nperm,mscB,q ,B1NsrcSC,2*Nq)]-1 ,Nw2Hex,2*Nq)];
            b_hex_snk_spin_weights(b, q%Nq, wnum, nsc, msc, nperm, 3) = hex_snk_spin_weights_T1_r3[index_3d(nsc,wnum,hex_perms[index_3d(nperm,mscB,q ,B1NsrcSC,2*Nq)]-1 ,Nw2Hex,2*Nq)];
         }
      }
      for (int msc=0; msc<B2NsrcSC; msc++) 
      for (int nsc=0; nsc<B1NsnkSC; nsc++) 
      for (int wnum=0; wnum< Nw2Hex; wnum++) {
         for (int q=0; q < 2*Nq; q++) {
            int b = (q-(q%Nq))/Nq;
            int nscB = nsc;
            b_hex_src_color_weights(b, q%Nq, wnum, msc, nsc, nperm, 0) = hex_snk_color_weights_A1[index_3d(msc,wnum,hex_perms[index_3d(nperm,nscB,q ,B1NsnkSC,2*Nq)]-1 ,Nw2Hex,2*Nq)];
            b_hex_src_spin_weights(b, q%Nq, wnum, msc, nsc, nperm, 0) = hex_snk_spin_weights_A1[index_3d(msc,wnum,hex_perms[index_3d(nperm,nscB,q ,B1NsnkSC,2*Nq)]-1 ,Nw2Hex,2*Nq)];
            b_hex_src_color_weights(b, q%Nq, wnum, msc, nsc, nperm, 1) = hex_snk_color_weights_T1_r1[index_3d(msc,wnum,hex_perms[index_3d(nperm,nscB,q ,B1NsnkSC,2*Nq)]-1 ,Nw2Hex,2*Nq)];
            b_hex_src_spin_weights(b, q%Nq, wnum, msc, nsc, nperm, 1) = hex_snk_spin_weights_T1_r1[index_3d(msc,wnum,hex_perms[index_3d(nperm,nscB,q ,B1NsnkSC,2*Nq)]-1 ,Nw2Hex,2*Nq)];
            b_hex_src_color_weights(b, q%Nq, wnum, msc, nsc, nperm, 2) = hex_snk_color_weights_T1_r2[index_3d(msc,wnum,hex_perms[index_3d(nperm,nscB,q ,B1NsnkSC,2*Nq)]-1 ,Nw2Hex,2*Nq)];
            b_hex_src_spin_weights(b, q%Nq, wnum, msc, nsc, nperm, 2) = hex_snk_spin_weights_T1_r2[index_3d(msc,wnum,hex_perms[index_3d(nperm,nscB,q ,B1NsnkSC,2*Nq)]-1 ,Nw2Hex,2*Nq)];
            b_hex_src_color_weights(b, q%Nq, wnum, msc, nsc, nperm, 3) = hex_snk_color_weights_T1_r3[index_3d(msc,wnum,hex_perms[index_3d(nperm,nscB,q ,B1NsnkSC,2*Nq)]-1 ,Nw2Hex,2*Nq)];
            b_hex_src_spin_weights(b, q%Nq, wnum, msc, nsc, nperm, 3) = hex_snk_spin_weights_T1_r3[index_3d(msc,wnum,hex_perms[index_3d(nperm,nscB,q ,B1NsnkSC,2*Nq)]-1 ,Nw2Hex,2*Nq)];
         }
      }
      for (int nsc=0; nsc<B2NsnkSC; nsc++) 
      for (int wnum=0; wnum< Nw2Hex; wnum++) {
         for (int q=0; q < 2*Nq; q++) {
            int b = (q-(q%Nq))/Nq;
            b_hex_hex_snk_color_weights(b, q%Nq, wnum, nsc, nperm, 0) = hex_snk_color_weights_A1[index_3d(nsc,wnum,hex_hex_perms[index_2d(nperm,q ,2*Nq)]-1 ,Nw2Hex,2*Nq)];
            b_hex_hex_snk_spin_weights(b, q%Nq, wnum, nsc, nperm, 0) = hex_snk_spin_weights_A1[index_3d(nsc,wnum,hex_hex_perms[index_2d(nperm,q ,2*Nq)]-1 ,Nw2Hex,2*Nq)];
            b_hex_hex_snk_color_weights(b, q%Nq, wnum, nsc, nperm, 1) = hex_snk_color_weights_T1_r1[index_3d(nsc,wnum,hex_hex_perms[index_2d(nperm,q ,2*Nq)]-1 ,Nw2Hex,2*Nq)];
            b_hex_hex_snk_spin_weights(b, q%Nq, wnum, nsc, nperm, 1) = hex_snk_spin_weights_T1_r1[index_3d(nsc,wnum,hex_hex_perms[index_2d(nperm,q ,2*Nq)]-1 ,Nw2Hex,2*Nq)];
            b_hex_hex_snk_color_weights(b, q%Nq, wnum, nsc, nperm, 2) = hex_snk_color_weights_T1_r2[index_3d(nsc,wnum,hex_hex_perms[index_2d(nperm,q ,2*Nq)]-1 ,Nw2Hex,2*Nq)];
            b_hex_hex_snk_spin_weights(b, q%Nq, wnum, nsc, nperm, 2) = hex_snk_spin_weights_T1_r2[index_3d(nsc,wnum,hex_hex_perms[index_2d(nperm,q ,2*Nq)]-1 ,Nw2Hex,2*Nq)];
            b_hex_hex_snk_color_weights(b, q%Nq, wnum, nsc, nperm, 3) = hex_snk_color_weights_T1_r3[index_3d(nsc,wnum,hex_hex_perms[index_2d(nperm,q ,2*Nq)]-1 ,Nw2Hex,2*Nq)];
            b_hex_hex_snk_spin_weights(b, q%Nq, wnum, nsc, nperm, 3) = hex_snk_spin_weights_T1_r3[index_3d(nsc,wnum,hex_hex_perms[index_2d(nperm,q ,2*Nq)]-1 ,Nw2Hex,2*Nq)];
         }
      }
   }
   for (int nsc=0; nsc<B2NsnkSC; nsc++) 
   for (int wnum=0; wnum< Nw2Hex; wnum++) {
      b_hex_snk_weights(wnum, nsc, 0) = hex_snk_weights_A1[index_2d(nsc,wnum ,Nw2Hex)];
      b_hex_snk_weights(wnum, nsc, 1) = hex_snk_weights_T1_r1[index_2d(nsc,wnum ,Nw2Hex)];
      b_hex_snk_weights(wnum, nsc, 2) = hex_snk_weights_T1_r2[index_2d(nsc,wnum ,Nw2Hex)];
      b_hex_snk_weights(wnum, nsc, 3) = hex_snk_weights_T1_r3[index_2d(nsc,wnum ,Nw2Hex)];
   }
   for (int msc=0; msc<B2NsrcSC; msc++) 
   for (int wnum=0; wnum< Nw2Hex; wnum++) {
      b_hex_src_weights(wnum, msc, 0) = hex_snk_weights_A1[index_2d(msc,wnum ,Nw2Hex)];
      b_hex_src_weights(wnum, msc, 1) = hex_snk_weights_T1_r1[index_2d(msc,wnum ,Nw2Hex)];
      b_hex_src_weights(wnum, msc, 2) = hex_snk_weights_T1_r2[index_2d(msc,wnum ,Nw2Hex)];
      b_hex_src_weights(wnum, msc, 3) = hex_snk_weights_T1_r3[index_2d(msc,wnum ,Nw2Hex)];
   }

   for (int rp=0; rp<B2Nrows; rp++)
      for (int m=0; m<NsrcTot; m++)
         for (int r=0; r<B2Nrows; r++)
            for (int n=0; n<NsnkTot; n++)
               for (int t=0; t<Lt; t++) 
                  for (int x=0; x<Vsnk/sites_per_rank; x++) {
                     b_C_r(n,r,m,rp,x,t) = 0.0;
                     b_C_i(n,r,m,rp,x,t) = 0.0;
                 } 

   if (rank == 0) {
   printf("prop 1 %4.9f + I %4.9f \n", b_B1_prop_r(0,0,0,0,0,0,0,0), b_B1_prop_i(0,0,0,0,0,0,0,0));
   printf("psi src 1 %4.9f + I %4.9f \n", b_B1_src_psi_r(0,0), b_B1_src_psi_i(0,0));
   printf("psi src 2 %4.9f + I %4.9f \n", b_B2_src_psi_r(0,0), b_B2_src_psi_i(0,0));
   printf("psi snk %4.9f + I %4.9f \n", b_snk_psi_r(0,0,0), b_snk_psi_i(0,0,0));
   if (B1NsnkSC > 0)
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
				    b_hex_sigs.raw_buffer(),
				    b_hex_hex_sigs.raw_buffer(),
				 b_src_color_weights.raw_buffer(),
				 b_src_spin_weights.raw_buffer(),
				 b_src_weights.raw_buffer(),
				    b_snk_b.raw_buffer(),
				    b_snk_color_weights.raw_buffer(),
				    b_snk_spin_weights.raw_buffer(),
				    b_snk_weights.raw_buffer(),
				    b_hex_snk_color_weights.raw_buffer(),
				    b_hex_snk_spin_weights.raw_buffer(),
				    b_hex_snk_weights.raw_buffer(),
				    b_hex_src_color_weights.raw_buffer(),
				    b_hex_src_spin_weights.raw_buffer(),
				    b_hex_src_weights.raw_buffer(),
				    b_hex_hex_snk_color_weights.raw_buffer(),
				    b_hex_hex_snk_spin_weights.raw_buffer());

   printf("done \n");


   if (rank == 0) {
   printf("BB-BB non-zero? %4.9e + I %4.9e \n", b_C_r(0,0,0,0,0,0), b_C_i(0,0,0,0,0,0) );
   printf("BB-BB non-zero? %4.9e + I %4.9e \n", b_C_r(0,1,0,0,0,0), b_C_i(0,1,0,0,0,0) );
   printf("BB-BB non-zero? %4.9e + I %4.9e \n", b_C_r(0,2,0,0,0,0), b_C_i(0,2,0,0,0,0) );
   printf("BB-BB non-zero? %4.9e + I %4.9e \n", b_C_r(0,3,0,0,0,0), b_C_i(0,3,0,0,0,0) );
   printf("H-BB non-zero? %4.9e + I %4.9e \n", b_C_r(0,0,B1NsrcSC*Nsrc,0,0,0), b_C_i(0,0,B1NsrcSC*Nsrc,0,0,0) );
   printf("H-BB non-zero? %4.9e + I %4.9e \n", b_C_r(0,1,B1NsrcSC*Nsrc,0,0,0), b_C_i(0,1,B1NsrcSC*Nsrc,0,0,0) );
   printf("H-BB non-zero? %4.9e + I %4.9e \n", b_C_r(0,2,B1NsrcSC*Nsrc,0,0,0), b_C_i(0,2,B1NsrcSC*Nsrc,0,0,0) );
   printf("H-BB non-zero? %4.9e + I %4.9e \n", b_C_r(0,3,B1NsrcSC*Nsrc,0,0,0), b_C_i(0,3,B1NsrcSC*Nsrc,0,0,0) );
   printf("H-H non-zero? %4.9e + I %4.9e \n", b_C_r(B1NsnkSC*Nsnk,0,B1NsrcSC*Nsrc,0,0,0), b_C_i(B1NsnkSC*Nsnk,0,B1NsrcSC*Nsrc,0,0,0) );
   printf("H-H non-zero? %4.9e + I %4.9e \n", b_C_r(B1NsnkSC*Nsnk,1,B1NsrcSC*Nsrc,0,0,0), b_C_i(B1NsnkSC*Nsnk,1,B1NsrcSC*Nsrc,0,0,0) );
   printf("H-H non-zero? %4.9e + I %4.9e \n", b_C_r(B1NsnkSC*Nsnk,2,B1NsrcSC*Nsrc,0,0,0), b_C_i(B1NsnkSC*Nsnk,2,B1NsrcSC*Nsrc,0,0,0) );
   printf("H-H non-zero? %4.9e + I %4.9e \n", b_C_r(B1NsnkSC*Nsnk,3,B1NsrcSC*Nsrc,0,0,0), b_C_i(B1NsnkSC*Nsnk,3,B1NsrcSC*Nsrc,0,0,0) );
   printf("H-H non-zero? %4.9e + I %4.9e \n", b_C_r(B1NsnkSC*Nsnk+B2NsnkSC*NsnkHex-1,0,B1NsrcSC*Nsrc+B2NsrcSC*NsrcHex-1,0,0,0), b_C_i(B1NsnkSC*Nsnk+B2NsnkSC*NsnkHex-1,0,B1NsrcSC*Nsrc+B2NsrcSC*NsrcHex-1,0,0,0) );
   printf("H-H non-zero? %4.9e + I %4.9e \n", b_C_r(B1NsnkSC*Nsnk+B2NsnkSC*NsnkHex-1,1,B1NsrcSC*Nsrc+B2NsrcSC*NsrcHex-1,0,0,0), b_C_i(B1NsnkSC*Nsnk+B2NsnkSC*NsnkHex-1,1,B1NsrcSC*Nsrc+B2NsrcSC*NsrcHex-1,0,0,0) );
   printf("H-H non-zero? %4.9e + I %4.9e \n", b_C_r(B1NsnkSC*Nsnk+B2NsnkSC*NsnkHex-1,2,B1NsrcSC*Nsrc+B2NsrcSC*NsrcHex-1,0,0,0), b_C_i(B1NsnkSC*Nsnk+B2NsnkSC*NsnkHex-1,2,B1NsrcSC*Nsrc+B2NsrcSC*NsrcHex-1,0,0,0) );
   printf("H-H non-zero? %4.9e + I %4.9e \n", b_C_r(B1NsnkSC*Nsnk+B2NsnkSC*NsnkHex-1,3,B1NsrcSC*Nsrc+B2NsrcSC*NsrcHex-1,0,0,0), b_C_i(B1NsnkSC*Nsnk+B2NsnkSC*NsnkHex-1,3,B1NsrcSC*Nsrc+B2NsrcSC*NsrcHex-1,0,0,0) );
   }

    // symmetrize and such
#ifdef WITH_MPI
   for (int rp=0; rp<B2Nrows; rp++)
      for (int m=0; m<NsrcTot; m++)
         for (int r=0; r<B2Nrows; r++)
            for (int n=0; n<NsnkTot; n++)
               for (int t=0; t<Lt; t++) {
                  double number0r;
                  double number0i;
                  double this_number0r = b_C_r(n,r,m,rp,rank,t);
                  double this_number0i = b_C_i(n,r,m,rp,rank,t);
                  MPI_Allreduce(&this_number0r, &number0r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                  MPI_Allreduce(&this_number0i, &number0i, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                  C_re[index_5d(rp,m,r,n,t, NsrcTot,B2Nrows,NsnkTot,Lt)] += number0r;
                  C_im[index_5d(rp,m,r,n,t, NsrcTot,B2Nrows,NsnkTot,Lt)] += number0i;
            }
#else
   for (int rp=0; rp<B2Nrows; rp++)
      for (int m=0; m<NsrcTot; m++)
         for (int r=0; r<B2Nrows; r++)
            for (int n=0; n<NsnkTot; n++)
               for (int t=0; t<Lt; t++)
                  for (int x=0; x<Vsnk; x++) {
                     double number0r = b_C_r(n,r,m,rp,x,t);
                     double number0i = b_C_i(n,r,m,rp,x,t);
                     C_re[index_5d(rp,m,r,n,t, NsrcTot,B2Nrows,NsnkTot,Lt)] += number0r;
                     C_im[index_5d(rp,m,r,n,t, NsrcTot,B2Nrows,NsnkTot,Lt)] += number0i;
                  }
#endif

   if (rank == 0) {
   for (int b=0; b<B2Nrows; b++) {
      printf("\n");
      for (int m=0; m<NsrcTot; m++)
         for (int n=0; n<NsnkTot; n++)
            for (int t=0; t<Lt; t++) {
                  printf("r=%d, m=%d, n=%d, t=%d: %4.1f + I (%4.1f) \n", b, m, n, t, C_re[index_5d(b,m,b,n,t, NsrcTot,B2Nrows,NsnkTot,Lt)],  C_im[index_5d(b,m,b,n,t, NsrcTot,B2Nrows,NsnkTot,Lt)]);
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

   int q, t, iC, iS, jC, jS, y, x, x1, x2, msc, nsc, m, n, k, wnum, nperm, b, r, rp;
   int iC1, iS1, iC2, iS2, jC1, jS1, jC2, jS2, kC1, kS1, kC2, kS2;

   int NsrcTot = B1NsrcSC*Nsrc+B2NsrcSC*NsrcHex;
   int NsnkTot = B1NsnkSC*Nsnk+B2NsnkSC*NsnkHex;

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
                           B1_prop_im[prop_index(q,t,jC,jS,iC,iS,y,x ,Nc,Ns,Vsrc,Vsnk,Lt)] = v3;
                           B2_prop_re[prop_index(q,t,jC,jS,iC,iS,y,x ,Nc,Ns,Vsrc,Vsnk,Lt)] = v2;
                           B2_prop_im[prop_index(q,t,jC,jS,iC,iS,y,x ,Nc,Ns,Vsrc,Vsnk,Lt)] = v4;
			   }
			   else {
                           if ((jC == iC) && (jS == iS)) {
                              B1_prop_re[prop_index(q,t,jC,jS,iC,iS,y,x ,Nc,Ns,Vsrc,Vsnk,Lt)] = 1/mq*cos(2*M_PI/6);
                              B1_prop_im[prop_index(q,t,jC,jS,iC,iS,y,x ,Nc,Ns,Vsrc,Vsnk,Lt)] = 1/mq*sin(2*M_PI/6);
                              B2_prop_re[prop_index(q,t,jC,jS,iC,iS,y,x ,Nc,Ns,Vsrc,Vsnk,Lt)] = 1/mq*cos(2*M_PI/6);
                              B2_prop_im[prop_index(q,t,jC,jS,iC,iS,y,x ,Nc,Ns,Vsrc,Vsnk,Lt)] = 1/mq*sin(2*M_PI/6);
                           }
                           else {
                              B1_prop_re[prop_index(q,t,jC,jS,iC,iS,y,x ,Nc,Ns,Vsrc,Vsnk,Lt)] = 0;
                              B1_prop_im[prop_index(q,t,jC,jS,iC,iS,y,x ,Nc,Ns,Vsrc,Vsnk,Lt)] = 0;
                              B2_prop_re[prop_index(q,t,jC,jS,iC,iS,y,x ,Nc,Ns,Vsrc,Vsnk,Lt)] = 0;
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
   double* t_snk_psi_re = (double *) malloc(sites_per_rank * Vsnk * NEntangled * sizeof (double));
   double* t_snk_psi_im = (double *) malloc(sites_per_rank * Vsnk * NEntangled * sizeof (double));
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
      for (int x_in = 0; x_in < sites_per_rank; x_in++)
         for (x2 = 0; x2 < Vsnk; x2++) {
            x1 = rank*sites_per_rank + x_in;
            //snk_psi_re[index_3d(x1,x2,n ,Vsnk,NEntangled)] = snk_psi_B1_re[index_2d(x1,n ,Nsnk)]*snk_psi_B2_re[index_2d(x2,n ,Nsnk)] - snk_psi_B1_im[index_2d(x1,n ,Nsnk)]*snk_psi_B2_im[index_2d(x2,n ,Nsnk)] + snk_psi_B1_re[index_2d(x2,n ,Nsnk)]*snk_psi_B2_re[index_2d(x1,n ,Nsnk)] - snk_psi_B1_im[index_2d(x2,n ,Nsnk)]*snk_psi_B2_im[index_2d(x1,n ,Nsnk)];// / Vsnk;
            //snk_psi_im[index_3d(x1,x2,n ,Vsnk,NEntangled)] = snk_psi_B1_re[index_2d(x1,n ,Nsnk)]*snk_psi_B2_im[index_2d(x2,n ,Nsnk)] + snk_psi_B1_im[index_2d(x1,n ,Nsnk)]*snk_psi_B2_re[index_2d(x2,n ,Nsnk)] + snk_psi_B1_re[index_2d(x2,n ,Nsnk)]*snk_psi_B2_im[index_2d(x1,n ,Nsnk)] + snk_psi_B1_im[index_2d(x2,n ,Nsnk)]*snk_psi_B2_re[index_2d(x1,n ,Nsnk)];// / Vsnk;
            snk_psi_re[index_3d(x_in,x2,n ,Vsnk,NEntangled)] = snk_psi_B1_re[index_2d(x1,n ,Nsnk)]*snk_psi_B2_re[index_2d(x2,n ,Nsnk)] - snk_psi_B1_im[index_2d(x1,n ,Nsnk)]*snk_psi_B2_im[index_2d(x2,n ,Nsnk)] + snk_psi_B1_re[index_2d(x2,n ,Nsnk)]*snk_psi_B2_re[index_2d(x1,n ,Nsnk)] - snk_psi_B1_im[index_2d(x2,n ,Nsnk)]*snk_psi_B2_im[index_2d(x1,n ,Nsnk)];// / Vsnk;
            snk_psi_im[index_3d(x_in,x2,n ,Vsnk,NEntangled)] = snk_psi_B1_re[index_2d(x1,n ,Nsnk)]*snk_psi_B2_im[index_2d(x2,n ,Nsnk)] + snk_psi_B1_im[index_2d(x1,n ,Nsnk)]*snk_psi_B2_re[index_2d(x2,n ,Nsnk)] + snk_psi_B1_re[index_2d(x2,n ,Nsnk)]*snk_psi_B2_im[index_2d(x1,n ,Nsnk)] + snk_psi_B1_im[index_2d(x2,n ,Nsnk)]*snk_psi_B2_re[index_2d(x1,n ,Nsnk)];// / Vsnk;
            //snk_psi_re[index_3d(x1,x2,n ,Vsnk,Nsnk)] = 1;// / Vsnk;
            //snk_psi_im[index_3d(x1,x2,n ,Vsnk,Nsnk)] = 0;// / Vsnk;
         } 
   for (n = 0; n < Nsnk; n++)
      for (x1 = 0; x1 < Vsnk; x1++)
         for (x2 = 0; x2 < Vsnk; x2++) {
            all_snk_psi_re[index_3d(x1,x2,n ,Vsnk,Nsnk)] = snk_psi_B1_re[index_2d(x1,n ,Nsnk)]*snk_psi_B2_re[index_2d(x2,n ,Nsnk)] - snk_psi_B1_im[index_2d(x1,n ,Nsnk)]*snk_psi_B2_im[index_2d(x2,n ,Nsnk)] + snk_psi_B1_re[index_2d(x2,n ,Nsnk)]*snk_psi_B2_re[index_2d(x1,n ,Nsnk)] - snk_psi_B1_im[index_2d(x2,n ,Nsnk)]*snk_psi_B2_im[index_2d(x1,n ,Nsnk)];// / Vsnk;
            all_snk_psi_im[index_3d(x1,x2,n ,Vsnk,Nsnk)] = snk_psi_B1_re[index_2d(x1,n ,Nsnk)]*snk_psi_B2_im[index_2d(x2,n ,Nsnk)] + snk_psi_B1_im[index_2d(x1,n ,Nsnk)]*snk_psi_B2_re[index_2d(x2,n ,Nsnk)] + snk_psi_B1_re[index_2d(x2,n ,Nsnk)]*snk_psi_B2_im[index_2d(x1,n ,Nsnk)] + snk_psi_B1_im[index_2d(x2,n ,Nsnk)]*snk_psi_B2_re[index_2d(x1,n ,Nsnk)];// / Vsnk;
         } 
   for (n = 0; n < NEntangled; n++)
      for (int x_in = 0; x_in < sites_per_rank; x_in++)
         for (x2 = 0; x2 < Vsnk; x2++) {
            x1 = rank*sites_per_rank + x_in;
            t_snk_psi_re[index_3d(x_in,x2,n ,Vsnk,NEntangled)] = snk_psi_B1_re[index_2d(x1,n ,Nsnk)]*snk_psi_B2_re[index_2d(x2,n ,Nsnk)] - snk_psi_B1_im[index_2d(x1,n ,Nsnk)]*snk_psi_B2_im[index_2d(x2,n ,Nsnk)] + snk_psi_B1_re[index_2d(x2,n ,Nsnk)]*snk_psi_B2_re[index_2d(x1,n ,Nsnk)] - snk_psi_B1_im[index_2d(x2,n ,Nsnk)]*snk_psi_B2_im[index_2d(x1,n ,Nsnk)];// / Vsnk;
            t_snk_psi_im[index_3d(x_in,x2,n ,Vsnk,NEntangled)] = snk_psi_B1_re[index_2d(x1,n ,Nsnk)]*snk_psi_B2_im[index_2d(x2,n ,Nsnk)] + snk_psi_B1_im[index_2d(x1,n ,Nsnk)]*snk_psi_B2_re[index_2d(x2,n ,Nsnk)] + snk_psi_B1_re[index_2d(x2,n ,Nsnk)]*snk_psi_B2_im[index_2d(x1,n ,Nsnk)] + snk_psi_B1_im[index_2d(x2,n ,Nsnk)]*snk_psi_B2_re[index_2d(x1,n ,Nsnk)];// / Vsnk;
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

   int* src_color_weights_r1 = (int *) malloc(B1NsrcSC * Nw * Nq * sizeof (int));
   int* src_color_weights_r2 = (int *) malloc(B1NsrcSC * Nw * Nq * sizeof (int));
   int* src_spin_weights_r1 = (int *) malloc(B1NsrcSC * Nw * Nq * sizeof (int));
   int* src_spin_weights_r2 = (int *) malloc(B1NsrcSC * Nw * Nq * sizeof (int));
   double* src_weights_r1 = (double *) malloc(B1NsrcSC * Nw * sizeof (double));
   double* src_weights_r2 = (double *) malloc(B1NsrcSC * Nw * sizeof (double));
   for (msc = 0; msc < B1NsrcSC; msc++)
   for (wnum = 0; wnum < Nw; wnum++) {
      for (q = 0; q < Nq; q++) {
         src_color_weights_r1[index_3d(msc,wnum,q ,Nw,Nq)] = src_color_weights_r1_P[wnum][q];
         src_color_weights_r2[index_3d(msc,wnum,q ,Nw,Nq)] = src_color_weights_r2_P[wnum][q];
         src_spin_weights_r1[index_3d(msc,wnum,q ,Nw,Nq)] = src_spin_weights_r1_P[wnum][q];
         src_spin_weights_r2[index_3d(msc,wnum,q ,Nw,Nq)] = src_spin_weights_r2_P[wnum][q];
      }
      src_weights_r1[index_2d(msc,wnum ,Nw)] = src_weights_r1_P[wnum];
      src_weights_r2[index_2d(msc,wnum ,Nw)] = src_weights_r2_P[wnum];
   }
   int* snk_color_weights_A1 = (int *) malloc(B2NsrcSC * Nw2Hex * 2*Nq * sizeof (int));
   int* snk_color_weights_T1_r1 = (int *) malloc(B2NsrcSC * Nw2Hex * 2*Nq * sizeof (int));
   int* snk_color_weights_T1_r2 = (int *) malloc(B2NsrcSC * Nw2Hex * 2*Nq * sizeof (int));
   int* snk_color_weights_T1_r3 = (int *) malloc(B2NsrcSC * Nw2Hex * 2*Nq * sizeof (int));
   int* snk_spin_weights_A1 = (int *) malloc(B2NsrcSC * Nw2Hex * 2*Nq * sizeof (int));
   int* snk_spin_weights_T1_r1 = (int *) malloc(B2NsrcSC * Nw2Hex * 2*Nq * sizeof (int));
   int* snk_spin_weights_T1_r2 = (int *) malloc(B2NsrcSC * Nw2Hex * 2*Nq * sizeof (int));
   int* snk_spin_weights_T1_r3 = (int *) malloc(B2NsrcSC * Nw2Hex * 2*Nq * sizeof (int));
   double* snk_weights_A1 = (double *) malloc(B2NsrcSC * Nw2Hex * sizeof (double));
   double* snk_weights_T1_r1 = (double *) malloc(B2NsrcSC * Nw2Hex * sizeof (double));
   double* snk_weights_T1_r2 = (double *) malloc(B2NsrcSC * Nw2Hex * sizeof (double));
   double* snk_weights_T1_r3 = (double *) malloc(B2NsrcSC * Nw2Hex * sizeof (double));

   //for (wnum = 0; wnum < Nw2Hex; wnum++) {
   int smallHex = 32;
   //int smallHex = 400;
   for (msc = 0; msc < B2NsrcSC; msc++)
   for (wnum = 0; wnum < smallHex; wnum++) {
      for (q = 0; q < 2*Nq; q++) {
         snk_color_weights_A1[index_3d(msc,wnum,q ,Nw2Hex,2*Nq)] = q % Nc;
         snk_color_weights_T1_r1[index_3d(msc,wnum,q ,Nw2Hex,2*Nq)] = q % Nc;
         snk_color_weights_T1_r2[index_3d(msc,wnum,q ,Nw2Hex,2*Nq)] = q % Nc;
         snk_color_weights_T1_r3[index_3d(msc,wnum,q ,Nw2Hex,2*Nq)] = q % Nc;
      }
      snk_spin_weights_A1[index_3d(msc,wnum,0 ,Nw2Hex,2*Nq)] = 0;
      snk_spin_weights_T1_r1[index_3d(msc,wnum,0 ,Nw2Hex,2*Nq)] = 0;
      snk_spin_weights_T1_r2[index_3d(msc,wnum,0 ,Nw2Hex,2*Nq)] = 0;
      snk_spin_weights_T1_r3[index_3d(msc,wnum,0 ,Nw2Hex,2*Nq)] = 1;
      snk_spin_weights_A1[index_3d(msc,wnum,1 ,Nw2Hex,2*Nq)] = 1;
      snk_spin_weights_T1_r1[index_3d(msc,wnum,1 ,Nw2Hex,2*Nq)] = 1;
      snk_spin_weights_T1_r2[index_3d(msc,wnum,1 ,Nw2Hex,2*Nq)] = 1;
      snk_spin_weights_T1_r3[index_3d(msc,wnum,1 ,Nw2Hex,2*Nq)] = 0;
      snk_spin_weights_A1[index_3d(msc,wnum,2 ,Nw2Hex,2*Nq)] = 0;
      snk_spin_weights_T1_r1[index_3d(msc,wnum,2 ,Nw2Hex,2*Nq)] = 0;
      snk_spin_weights_T1_r2[index_3d(msc,wnum,2 ,Nw2Hex,2*Nq)] = 0;
      snk_spin_weights_T1_r3[index_3d(msc,wnum,2 ,Nw2Hex,2*Nq)] = 1;
      snk_spin_weights_A1[index_3d(msc,wnum,3 ,Nw2Hex,2*Nq)] = 1;
      snk_spin_weights_T1_r1[index_3d(msc,wnum,3 ,Nw2Hex,2*Nq)] = 0;
      snk_spin_weights_T1_r2[index_3d(msc,wnum,3 ,Nw2Hex,2*Nq)] = 1;
      snk_spin_weights_T1_r3[index_3d(msc,wnum,3 ,Nw2Hex,2*Nq)] = 1;
      snk_spin_weights_A1[index_3d(msc,wnum,4 ,Nw2Hex,2*Nq)] = 0;
      snk_spin_weights_T1_r1[index_3d(msc,wnum,4 ,Nw2Hex,2*Nq)] = 1;
      snk_spin_weights_T1_r2[index_3d(msc,wnum,4 ,Nw2Hex,2*Nq)] = 0;
      snk_spin_weights_T1_r3[index_3d(msc,wnum,4 ,Nw2Hex,2*Nq)] = 0;
      snk_spin_weights_A1[index_3d(msc,wnum,5 ,Nw2Hex,2*Nq)] = 1;
      snk_spin_weights_T1_r1[index_3d(msc,wnum,5 ,Nw2Hex,2*Nq)] = 0;
      snk_spin_weights_T1_r2[index_3d(msc,wnum,5 ,Nw2Hex,2*Nq)] = 1;
      snk_spin_weights_T1_r3[index_3d(msc,wnum,5 ,Nw2Hex,2*Nq)] = 1;
      snk_weights_A1[index_2d(msc,wnum ,Nw2Hex)] = -1.0/sqrt(2);
      snk_weights_T1_r1[index_2d(msc,wnum ,Nw2Hex)] = 1.0;
      snk_weights_T1_r2[index_2d(msc,wnum ,Nw2Hex)] = -1.0/sqrt(2);
      snk_weights_T1_r3[index_2d(msc,wnum ,Nw2Hex)] = 1.0;
   }
   for (msc = 0; msc < B2NsrcSC; msc++)
   for (wnum = smallHex; wnum < Nw2Hex; wnum++) {
      for (q = 0; q < 2*Nq; q++) {
         snk_color_weights_A1[index_3d(msc,wnum,q ,Nw2Hex,2*Nq)] = q % Nc;
         snk_color_weights_T1_r1[index_3d(msc,wnum,q ,Nw2Hex,2*Nq)] = q % Nc;
         snk_color_weights_T1_r2[index_3d(msc,wnum,q ,Nw2Hex,2*Nq)] = q % Nc;
         snk_color_weights_T1_r3[index_3d(msc,wnum,q ,Nw2Hex,2*Nq)] = q % Nc;
      }
      snk_spin_weights_A1[index_3d(msc,wnum,0 ,Nw2Hex,2*Nq)] = 0;
      snk_spin_weights_T1_r1[index_3d(msc,wnum,0 ,Nw2Hex,2*Nq)] = 0;
      snk_spin_weights_T1_r2[index_3d(msc,wnum,0 ,Nw2Hex,2*Nq)] = 0;
      snk_spin_weights_T1_r3[index_3d(msc,wnum,0 ,Nw2Hex,2*Nq)] = 1;
      snk_spin_weights_A1[index_3d(msc,wnum,1 ,Nw2Hex,2*Nq)] = 1;
      snk_spin_weights_T1_r1[index_3d(msc,wnum,1 ,Nw2Hex,2*Nq)] = 1;
      snk_spin_weights_T1_r2[index_3d(msc,wnum,1 ,Nw2Hex,2*Nq)] = 1;
      snk_spin_weights_T1_r3[index_3d(msc,wnum,1 ,Nw2Hex,2*Nq)] = 0;
      snk_spin_weights_A1[index_3d(msc,wnum,2 ,Nw2Hex,2*Nq)] = 0;
      snk_spin_weights_T1_r1[index_3d(msc,wnum,2 ,Nw2Hex,2*Nq)] = 0;
      snk_spin_weights_T1_r2[index_3d(msc,wnum,2 ,Nw2Hex,2*Nq)] = 0;
      snk_spin_weights_T1_r3[index_3d(msc,wnum,2 ,Nw2Hex,2*Nq)] = 1;
      snk_spin_weights_A1[index_3d(msc,wnum,3 ,Nw2Hex,2*Nq)] = 1;
      snk_spin_weights_T1_r1[index_3d(msc,wnum,3 ,Nw2Hex,2*Nq)] = 0;
      snk_spin_weights_T1_r2[index_3d(msc,wnum,3 ,Nw2Hex,2*Nq)] = 1;
      snk_spin_weights_T1_r3[index_3d(msc,wnum,3 ,Nw2Hex,2*Nq)] = 1;
      snk_spin_weights_A1[index_3d(msc,wnum,4 ,Nw2Hex,2*Nq)] = 0;
      snk_spin_weights_T1_r1[index_3d(msc,wnum,4 ,Nw2Hex,2*Nq)] = 1;
      snk_spin_weights_T1_r2[index_3d(msc,wnum,4 ,Nw2Hex,2*Nq)] = 0;
      snk_spin_weights_T1_r3[index_3d(msc,wnum,4 ,Nw2Hex,2*Nq)] = 0;
      snk_spin_weights_A1[index_3d(msc,wnum,5 ,Nw2Hex,2*Nq)] = 1;
      snk_spin_weights_T1_r1[index_3d(msc,wnum,5 ,Nw2Hex,2*Nq)] = 0;
      snk_spin_weights_T1_r2[index_3d(msc,wnum,5 ,Nw2Hex,2*Nq)] = 1;
      snk_spin_weights_T1_r3[index_3d(msc,wnum,5 ,Nw2Hex,2*Nq)] = 1;
      snk_weights_A1[index_2d(msc,wnum ,Nw2Hex)] = 0.0;
      snk_weights_T1_r1[index_2d(msc,wnum ,Nw2Hex)] = 0.0;
      snk_weights_T1_r2[index_2d(msc,wnum ,Nw2Hex)] = 0.0;
      snk_weights_T1_r3[index_2d(msc,wnum ,Nw2Hex)] = 0.0;
   } 
   // Permutations
   int perms_array[36][6] = { {1,2,3,4,5,6}, {1, 4, 3, 2, 5, 6}, {1, 6, 3, 2, 5, 4}, {1, 2, 3, 6, 5, 4}, {1, 4, 3, 6, 5, 2}, {1, 6, 3, 4, 5, 2}, {3, 2, 1, 4, 5, 6}, {3, 4, 1, 2, 5, 6}, {3, 6, 1, 2, 5, 4}, {3, 2, 1, 6, 5, 4}, {3, 4, 1, 6, 5, 2}, {3, 6, 1, 4, 5, 2}, {5, 2, 1, 4, 3, 6}, {5, 4, 1, 2, 3, 6}, {5, 6, 1, 2, 3, 4}, {5, 2, 1, 6, 3, 4}, {5, 4, 1, 6, 3, 2}, {5, 6, 1, 4, 3, 2}, {1, 2, 5, 4, 3, 6}, {1, 4, 5, 2, 3, 6}, {1, 6, 5, 2, 3, 4}, {1, 2, 5, 6, 3, 4}, {1, 4, 5, 6, 3, 2}, {1, 6, 5, 4, 3, 2}, {3, 2, 5, 4, 1, 6}, {3, 4, 5, 2, 1, 6}, {3, 6, 5, 2, 1, 4}, {3, 2, 5, 6, 1, 4}, {3, 4, 5, 6, 1, 2}, {3, 6, 5, 4, 1, 2}, {5, 2, 3, 4, 1, 6}, {5, 4, 3, 2, 1, 6}, {5, 6, 3, 2, 1, 4}, {5, 2, 3, 6, 1, 4}, {5, 4, 3, 6, 1, 2}, {5, 6, 3, 4, 1, 2} };
   int sigs_array[36] = {1,-1,1,-1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,1,-1,1,-1,1};
   int* perms = (int *) malloc(Nperms * B1NsrcSC * B1NsnkSC * 2*Nq * sizeof (int));
   int* hex_perms = (int *) malloc(Nperms * B1NsrcSC * 2*Nq * sizeof (int));
   int* hex_perms_snk = (int *) malloc(Nperms * B1NsnkSC * 2*Nq * sizeof (int));
   int* hex_hex_perms = (int *) malloc(Nperms * 2*Nq * sizeof (int));
   int* sigs = (int *) malloc(Nperms * B1NsrcSC * B1NsnkSC * sizeof (int));
   int* hex_sigs = (int *) malloc(Nperms * B1NsrcSC * sizeof (int));
   int* hex_sigs_snk = (int *) malloc(Nperms * B1NsnkSC * sizeof (int));
   int hex_hex_sigs[Nperms];
   int permnum = 0;
   for (int i = 0; i < Nperms; i++) {

      /*if (perms_array[i][0] > perms_array[i][2]) {
         continue;
      }
      else if (perms_array[i][3] > perms_array[i][5]) {
         continue;
      } 
      else {  */
         for (int q = 0; q < 2*Nq; q++) {
            hex_hex_perms[index_2d(permnum,q ,2*Nq)] = perms_array[i][q];
         }
         hex_hex_sigs[permnum] = sigs_array[i];

         for (int msc = 0; msc < B1NsrcSC; msc++) {
         for (int q = 0; q < 2*Nq; q++) {
            hex_perms[index_3d(permnum,msc,q ,B1NsrcSC,2*Nq)] = perms_array[i][q];
         }
         hex_sigs[index_2d(permnum,msc ,B1NsrcSC)] = sigs_array[i];
         }

         for (int nsc = 0; nsc < B1NsnkSC; nsc++) {
         for (int q = 0; q < 2*Nq; q++) {
            hex_perms_snk[index_3d(permnum,nsc,q ,B1NsnkSC,2*Nq)] = perms_array[i][q];
         }
         hex_sigs_snk[index_2d(permnum,nsc ,B1NsnkSC)] = sigs_array[i];
         }

         for (int msc = 0; msc < B1NsrcSC; msc++) {
         for (int nsc = 0; nsc < B1NsnkSC; nsc++) {
         for (int q = 0; q < 2*Nq; q++) {
            perms[index_4d(permnum,msc,nsc,q ,B1NsrcSC,B1NsnkSC,2*Nq)] = perms_array[i][q];
         }
         sigs[index_3d(permnum,msc,nsc ,B1NsrcSC,B1NsnkSC)] = sigs_array[i];
         }
         }

         permnum += 1;
      //}
   }
   int* BB_pairs_src = (int *) malloc(B1NsrcSC * 2 * sizeof(int));
   int* BB_pairs_snk = (int *) malloc(B1NsnkSC * 2 * sizeof(int));
   for (msc=0; msc < B1NsrcSC; msc++) {
      for (b = 0; b < 2; b++) {
         BB_pairs_src[index_2d(msc,b ,2)] = msc+1;
      }
   }
   for (nsc=0; nsc < B1NsnkSC; nsc++) {
      for (b = 0; b < 2; b++) {
         BB_pairs_snk[index_2d(nsc,b ,2)] = nsc+1;
      }
   }

   // Correlators
   double* C_re = (double *) malloc(B2Nrows * (Nsrc+NsrcHex) * (Nsnk+NsnkHex) * Lt * sizeof (double));
   double* C_im = (double *) malloc(B2Nrows * (Nsrc+NsrcHex) * (Nsnk+NsnkHex) * Lt * sizeof (double));
   //double* C_re = (double *) malloc(B2Nrows * B2Nrows * (Nsrc+NsrcHex) * (Nsnk+NsnkHex) * Lt * sizeof (double));
   //double* C_im = (double *) malloc(B2Nrows * B2Nrows * (Nsrc+NsrcHex) * (Nsnk+NsnkHex) * Lt * sizeof (double));
   double* t_C_re = (double *) malloc(B2Nrows * B2Nrows * NsrcTot * NsnkTot * Lt * sizeof (double));
   double* t_C_im = (double *) malloc(B2Nrows * B2Nrows * NsrcTot * NsnkTot * Lt * sizeof (double));
   for (rp=0; rp<B2Nrows; rp++)
      for (m=0; m<Nsrc+NsrcHex; m++)
         for (r=0; r<B2Nrows; r++)
            for (n=0; n<Nsnk+NsnkHex; n++)
               for (t=0; t<Lt; t++) {
                  C_re[index_4d(rp,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Lt)] = 0.0;
                  C_im[index_4d(rp,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Lt)] = 0.0;
               }
   for (rp=0; rp<B2Nrows; rp++)
      for (m=0; m<NsrcTot; m++)
         for (r=0; r<B2Nrows; r++)
            for (n=0; n<NsnkTot; n++)
               for (t=0; t<Lt; t++) {
                  t_C_re[index_5d(rp,m,r,n,t, NsrcTot,B2Nrows,NsnkTot,Lt)] = 0.0;
                  t_C_im[index_5d(rp,m,r,n,t, NsrcTot,B2Nrows,NsnkTot,Lt)] = 0.0;
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
           hex_perms,
           hex_sigs,
           hex_perms_snk,
           hex_sigs_snk,
           hex_hex_perms,
           hex_hex_sigs,
           BB_pairs_src, 
           BB_pairs_snk, 
           src_psi_B1_re, 
           src_psi_B1_im, 
           src_psi_B2_re, 
           src_psi_B2_im, 
           t_snk_psi_re,
           t_snk_psi_im, 
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

      make_two_nucleon_2pt(C_re, C_im, B1_prop_re, B1_prop_im, B2_prop_re, B2_prop_im, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, snk_color_weights_A1, snk_spin_weights_A1, snk_weights_A1, snk_color_weights_T1_r1, snk_spin_weights_T1_r1, snk_weights_T1_r1, snk_color_weights_T1_r2, snk_spin_weights_T1_r2, snk_weights_T1_r2, snk_color_weights_T1_r3, snk_spin_weights_T1_r3, snk_weights_T1_r3, hex_hex_perms, hex_hex_sigs, src_psi_B1_re, src_psi_B1_im, src_psi_B2_re, src_psi_B2_im, all_snk_psi_re, all_snk_psi_im, snk_psi_B1_re, snk_psi_B1_im, snk_psi_B2_re, snk_psi_B2_im, hex_src_psi_re, hex_src_psi_im, hex_snk_psi_re, hex_snk_psi_im, space_symmetric, snk_entangled, Nc, Ns, Vsrc, Vsnk, Lt, Nw, Nw2Hex, Nq, Nsrc, Nsnk, NsrcHex, NsnkHex, Nperms);
           

      make_two_nucleon_2pt(C_re, C_im, B1_prop_re, B1_prop_im, B2_prop_re, B2_prop_im, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, snk_color_weights_A1, snk_spin_weights_A1, snk_weights_A1, snk_color_weights_T1_r1, snk_spin_weights_T1_r1, snk_weights_T1_r1, snk_color_weights_T1_r2, snk_spin_weights_T1_r2, snk_weights_T1_r2, snk_color_weights_T1_r3, snk_spin_weights_T1_r3, snk_weights_T1_r3, hex_hex_perms, hex_hex_sigs, src_psi_B2_re, src_psi_B2_im, src_psi_B1_re, src_psi_B1_im, all_snk_psi_re, all_snk_psi_im, snk_psi_B2_re, snk_psi_B2_im, snk_psi_B1_re, snk_psi_B1_im, hex_src_psi_re, hex_src_psi_im, hex_snk_psi_re, hex_snk_psi_im, space_symmetric, snk_entangled, Nc, Ns, Vsrc, Vsnk, Lt, Nw, Nw2Hex, Nq, Nsrc, Nsnk, NsrcHex, NsnkHex, Nperms);

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
      for (m=0; m<Nsrc; m++)
      for (msc=0; msc<B1NsrcSC; msc++)
         for (n=0; n<Nsnk; n++)
         for (nsc=0; nsc<B1NsnkSC; nsc++)
//            for (r=0; r<B2Nrows; r++)
              for (t=0; t<Lt; t++) {
                 if ((std::abs(C_re[index_4d(rp,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Lt)] - t_C_re[index_5d(rp,msc*Nsrc+m,rp,nsc*Nsnk+n,t, NsrcTot,B2Nrows,NsnkTot,Lt)]) >= 0.01*Vsnk*Vsnk) ||
	               (std::abs(C_im[index_4d(rp,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Lt)] -  t_C_im[index_5d(rp,msc*Nsrc+m,rp,nsc*Nsnk+n,t, NsrcTot,B2Nrows,NsnkTot,Lt)]) >= 0.01*Vsnk*Vsnk))
	            {
                  printf("BB_BB rp=%d, msc=%d, m=%d, nsc=%d, n=%d, t=%d: %4.1f + I (%4.1f) vs %4.1f + I (%4.1f) \n", rp, msc, m, nsc, n, t, C_re[index_4d(rp,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Lt)], C_im[index_4d(rp,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Lt)],  t_C_re[index_5d(rp,msc*Nsrc+m,rp,nsc*Nsnk+n,t, NsrcTot,B2Nrows,NsnkTot,Lt)],  t_C_im[index_5d(rp,msc*Nsrc+m,rp,nsc*Nsnk+n,t, NsrcTot,B2Nrows,NsnkTot,Lt)]);
		            std::cout << "Error: different computed values for C_r or C_i!" << std::endl;
		            exit(1);
	            }
            }
   }
   if (rank == 0)
   printf("cleared BB-BB \n");
   for (rp=0; rp<B2Nrows; rp++) {
      for (msc=0; msc<B1NsrcSC; msc++)
      for (m=0; m<Nsrc; m++)
         for (nsc=0; nsc<B2NsrcSC; nsc++)
         for (n=0; n<NsnkHex; n++)
//            for (r=0; r<B2Nrows; r++)
              for (t=0; t<Lt; t++) {
                 if ((std::abs(C_re[index_4d(rp,m,Nsnk+n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Lt)] - t_C_re[index_5d(rp,m,rp,Nsnk*B1NsnkSC+n,t, NsrcTot,B2Nrows,NsnkTot,Lt)]) >= 0.01*Vsnk*Vsnk) ||
	               (std::abs(C_im[index_4d(rp,m,Nsnk+n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Lt)] -  t_C_im[index_5d(rp,m,rp,Nsnk*B1NsnkSC+n,t, NsrcTot,B2Nrows,NsnkTot,Lt)]) >= 0.01*Vsnk*Vsnk))
	            {
                  printf("BB_H rp=%d, msc=%d, m=%d, nsc=%d, n=%d, t=%d: %4.1f + I (%4.1f) vs %4.1f + I (%4.1f) \n", rp, msc, m, nsc, n, t, C_re[index_4d(rp,m,Nsnk+n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Lt)], C_im[index_4d(rp,m,Nsnk+n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Lt)],  t_C_re[index_5d(rp,msc*Nsrc+m,rp,Nsnk*B1NsnkSC+nsc*NsnkHex+n,t, NsrcTot,B2Nrows,NsnkTot,Lt)],  t_C_im[index_5d(rp,msc*Nsrc+m,rp,Nsnk*B1NsnkSC+nsc*NsnkHex+n,t, NsrcTot,B2Nrows,NsnkTot,Lt)]);
		            std::cout << "Error: different computed values for C_r or C_i!" << std::endl;
		            exit(1);
	            }
            }
   }
   if (rank == 0)
   printf("cleared BB-H \n");
   for (rp=0; rp<B2Nrows; rp++) {
      for (msc=0; msc<B2NsrcSC; msc++)
      for (m=0; m<NsrcHex; m++)
         for (nsc=0; nsc<B1NsrcSC; nsc++)
         for (n=0; n<Nsnk; n++)
//            for (r=0; r<B2Nrows; r++)
              for (t=0; t<Lt; t++) {
                 if ((std::abs(C_re[index_4d(rp,Nsrc+m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Lt)] - t_C_re[index_5d(rp,Nsrc*B1NsrcSC+msc*NsrcHex+m,rp,nsc*Nsnk+n,t, NsrcTot,B2Nrows,NsnkTot,Lt)]) >= 0.01*Vsnk*Vsnk) ||
	               (std::abs(C_im[index_4d(rp,Nsrc+m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Lt)] -  t_C_im[index_5d(rp,Nsrc*B1NsrcSC+msc*NsrcHex+m,rp,nsc*Nsnk+n,t, NsrcTot,B2Nrows,NsnkTot,Lt)]) >= 0.01*Vsnk*Vsnk))
	            {
                  printf("H_BB rp=%d, msc=%d, m=%d, nsc=%d, n=%d, t=%d: %4.1f + I (%4.1f) vs %4.1f + I (%4.1f) \n", rp, msc, m, nsc, n, t, C_re[index_4d(rp,Nsrc+m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Lt)], C_im[index_4d(rp,Nsrc+m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Lt)],  t_C_re[index_5d(rp,Nsrc*B1NsrcSC+msc*NsrcHex+m,rp,nsc*Nsnk+n,t, NsrcTot,B2Nrows,NsnkTot,Lt)],  t_C_im[index_5d(rp,Nsrc*B1NsrcSC+msc*NsrcHex+m,rp,nsc*Nsnk+n,t, NsrcTot,B2Nrows,NsnkTot,Lt)]);
		            std::cout << "Error: different computed values for C_r or C_i!" << std::endl;
		            exit(1);
	            }
            }
   }
   if (rank == 0)
   printf("cleared H-BB \n");
   for (rp=0; rp<B2Nrows; rp++) {
      for (nsc=0; nsc<B2NsrcSC; nsc++)
      for (m=0; m<NsrcHex; m++)
         for (msc=0; msc<=nsc; msc++)
         for (n=0; n<NsnkHex; n++)
//            for (r=0; r<B2Nrows; r++)
              for (t=0; t<Lt; t++) {
                 if ((std::abs(C_re[index_4d(rp,Nsrc+m,Nsnk+n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Lt)] - t_C_re[index_5d(rp,Nsrc*B1NsrcSC+msc*NsrcHex+m,rp,Nsnk*B1NsnkSC+nsc*NsnkHex+n,t, NsrcTot,B2Nrows,NsnkTot,Lt)]) >= 0.01*Vsnk*Vsnk) ||
	               (std::abs(C_im[index_4d(rp,Nsrc+m,Nsnk+n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Lt)] -  t_C_im[index_5d(rp,Nsrc*B1NsrcSC+msc*NsrcHex+m,rp,Nsnk*B1NsnkSC+nsc*NsnkHex+n,t, NsrcTot,B2Nrows,NsnkTot,Lt)]) >= 0.01*Vsnk*Vsnk))
	            {
                  printf("H_H rp=%d, msc=%d, m=%d, nsc=%d, n=%d, t=%d: %4.1f + I (%4.1f) vs %4.1f + I (%4.1f) \n", rp, msc, m, nsc, n, t, C_re[index_4d(rp,Nsrc+m,Nsnk+n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Lt)], C_im[index_4d(rp,Nsrc+m,Nsnk+n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Lt)],  t_C_re[index_5d(rp,Nsrc*B1NsrcSC+msc*NsrcHex+m,rp,Nsnk*B1NsnkSC+nsc*NsnkHex+n,t, NsrcTot,B2Nrows,NsnkTot,Lt)],  t_C_im[index_5d(rp,Nsrc*B1NsrcSC+msc*NsrcHex+m,rp,Nsnk*B1NsnkSC+nsc*NsnkHex+n,t, NsrcTot,B2Nrows,NsnkTot,Lt)]);
		            std::cout << "Error: different computed values for C_r or C_i!" << std::endl;
		            exit(1);
	            }
            }
   }
   if (rank == 0)
   printf("cleared H-H \n");

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
