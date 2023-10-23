#include "Halide.h"
#include <tiramisu/utils.h>
#include <tiramisu/mpi_comm.h>
#include <cstdlib>
#include <iostream>
#include <complex>
#include <cmath>

#ifdef __cplusplus
extern "C" {
#endif

#include "gpu_tiramisu_make_fused_identical_baryon_blocks_correlator_wrapper.h"
#include "gpu_tiramisu_make_fused_baryon_blocks_correlator_ref.cpp"

#define RUN_REFERENCE 1
#define RUN_CHECK 1
int nb_tests = 1;
int randommode = 0;

const int Nc_f = Nc;
const int Ns_f= Ns;
const int Vsrc_f = Vsrc;
const int Vsnk_f = Vsnk;
const int Nt_f = Lt;
const int Nw_f = Nw;
const int Nq_f = Nq;
const int Nsrc_f = NsrcHex;
const int Nsnk_f = NsnkHex;

void gpu_tiramisu_make_identical_nucleon_2pt(double* C_re,
    double* C_im,
     double* B1_prop_re, 
     double* B1_prop_im, 
     int *src_color_weights_r1,
     int *src_spin_weights_r1,
     double *src_weights_r1,
     int *src_color_weights_r2,
     int *src_spin_weights_r2,
     double *src_weights_r2,
     int *perms,
     int *sigs,
     double* src_psi_B1_re, 
     double* src_psi_B1_im, 
     double* snk_psi_B1_re, 
     double* snk_psi_B1_im)
{

   int q, t, iC, iS, jC, jS, y, x, x1, x2, m, n, k, wnum, b, rp, r;
   int iC1, iS1, iC2, iS2, jC1, jS1, jC2, jS2, kC1, kS1, kC2, kS2;
    int rank = 0;
#ifdef WITH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

  // printf("hi I'm rank %d \n", rank);

    if (rank == 0) {
    long mega = 1024*1024;
    std::cout << "Array sizes" << "\n";
    std::cout << "Prop:" <<  "\n";
    std::cout << "	Max index size = " << Vsnk*Vsrc*Nc*Ns*Nc*Ns*Lt <<  "\n";
    std::cout << "	Array size = " << Vsnk*Vsrc*Nc*Ns*Nc*Ns*Lt*sizeof(std::complex<double>)/mega << " Mega bytes" << "\n";
    std::cout << "Q, O & P:" <<  "\n";
    std::cout << "	Max index size = " << Vsnk*Vsrc*Nc*Ns*Nc*Ns <<  "\n";
    std::cout << "	Array size = " << Vsnk*Vsrc*Nc*Ns*Nc*Ns*sizeof(std::complex<double>)/mega << " Mega bytes" <<  "\n";

    long kilo = 1024;
    std::cout << "Blocal:" <<  "\n";
    std::cout << "	Max index size = " << Vsnk*NsrcHex*Nc*Ns*Nc*Ns*Nc*Ns <<  "\n";
    std::cout << "	Array size = " << Vsnk*NsrcHex*Nc*Ns*Nc*Ns*Nc*Ns*sizeof(std::complex<double>)/kilo << " kilo bytes" <<  "\n";
    std::cout << "Blocal, Bsingle, Bdouble:" <<  "\n";
    std::cout << "	Max index size = " << Nc*Ns*Nc*Ns*Nc*Ns <<  "\n";
    std::cout << "	Array size = " << Nc*Ns*Nc*Ns*Nc*Ns*sizeof(std::complex<double>)/kilo << " kilo bytes" <<  "\n";
    std::cout << "\n";
    }

   // Halide buffers
   Halide::Buffer<double> b_C_r(NsnkHex, B1Nrows, NsrcHex, B1Nrows, Lt, "C_r");
   Halide::Buffer<double> b_C_i(NsnkHex, B1Nrows, NsrcHex, B1Nrows, Lt, "C_i");

   Halide::Buffer<double> b_C_r_full(NsnkHex, B1Nrows, NsrcHex, B1Nrows, sites_per_rank, Vsnk/sites_per_rank, Lt, "C_r_full");
   Halide::Buffer<double> b_C_i_full(NsnkHex, B1Nrows, NsrcHex, B1Nrows, sites_per_rank, Vsnk/sites_per_rank, Lt, "C_i_full");

   Halide::Buffer<int> b_src_color_weights(Nq, Nw, B1Nrows, "src_color_weights");
   Halide::Buffer<int> b_src_spin_weights(Nq, Nw, B1Nrows, "src_spin_weights");
   Halide::Buffer<double> b_src_weights(Nw, B1Nrows, "src_weights");

   Halide::Buffer<int> b_src_spins(B1Nrows, "src_spins");
   Halide::Buffer<int> b_snk_color_weights(Nq, Nw, B1Nperms, B1Nrows, "snk_color_weights");
   Halide::Buffer<int> b_snk_spin_weights(Nq, Nw, B1Nperms, B1Nrows, "snk_spin_weights");
   Halide::Buffer<double> b_snk_weights(Nw, B1Nrows, "snk_weights");

    // prop
    Halide::Buffer<double> b_B1_prop_r((double *)B1_prop_re, {Vsrc, Vsnk, Ns, Nc, Ns, Nc, Lt});
    Halide::Buffer<double> b_B1_prop_i((double *)B1_prop_im, {Vsrc, Vsnk, Ns, Nc, Ns, Nc, Lt});

    if (rank == 0)
   printf("prop elem %4.9f \n", b_B1_prop_r(0,0,0,0,0,0,0));

    // psi
    Halide::Buffer<double> b_B1_src_psi_r((double *)src_psi_B1_re, {NsrcHex, Vsrc});
    Halide::Buffer<double> b_B1_src_psi_i((double *)src_psi_B1_im, {NsrcHex, Vsrc});
    Halide::Buffer<double> b_B1_snk_psi_r((double *)snk_psi_B1_re, {NsnkHex, sites_per_rank, Vsnk/sites_per_rank});
    Halide::Buffer<double> b_B1_snk_psi_i((double *)snk_psi_B1_im, {NsnkHex, sites_per_rank, Vsnk/sites_per_rank});

   Halide::Buffer<int> b_sigs((int *)sigs, {B1Nperms});

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
   b_src_spins(0) = 1;
   b_src_spins(1) = 2;
   for (int nperm=0; nperm<B1Nperms; nperm++) {
      int snk_1 = perms[index_2d(nperm,0 ,Nq)] - 1;
      int snk_2 = perms[index_2d(nperm,1 ,Nq)] - 1;
      int snk_3 = perms[index_2d(nperm,2 ,Nq)] - 1;
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

         b_snk_color_weights(0, wnum, nperm, 0) = snk_color_weights_r1[index_2d(wnum,snk_1 ,Nq)];
         b_snk_spin_weights(0, wnum, nperm, 0) = snk_spin_weights_r1[index_2d(wnum,snk_1 ,Nq)];
         b_snk_color_weights(1, wnum, nperm, 0) = snk_color_weights_r1[index_2d(wnum,snk_2 ,Nq)];
         b_snk_spin_weights(1, wnum, nperm, 0) = snk_spin_weights_r1[index_2d(wnum,snk_2 ,Nq)];
         b_snk_color_weights(2, wnum, nperm, 0) = snk_color_weights_r1[index_2d(wnum,snk_3 ,Nq)];
         b_snk_spin_weights(2, wnum, nperm, 0) = snk_spin_weights_r1[index_2d(wnum,snk_3 ,Nq)];

         b_snk_color_weights(0, wnum, nperm, 1) = snk_color_weights_r2[index_2d(wnum,snk_1 ,Nq)];
         b_snk_spin_weights(0, wnum, nperm, 1) = snk_spin_weights_r2[index_2d(wnum,snk_1 ,Nq)];
         b_snk_color_weights(1, wnum, nperm, 1) = snk_color_weights_r2[index_2d(wnum,snk_2 ,Nq)];
         b_snk_spin_weights(1, wnum, nperm, 1) = snk_spin_weights_r2[index_2d(wnum,snk_2 ,Nq)];
         b_snk_color_weights(2, wnum, nperm, 1) = snk_color_weights_r2[index_2d(wnum,snk_3 ,Nq)];
         b_snk_spin_weights(2, wnum, nperm, 1) = snk_spin_weights_r2[index_2d(wnum,snk_3 ,Nq)];
      }
   }

   for (int rp=0; rp<B1Nrows; rp++)
      for (int m=0; m<NsrcHex; m++)
         for (int r=0; r<B1Nrows; r++)
            for (int n=0; n<NsnkHex; n++)
               for (int t=0; t<Lt; t++) 
                     {
                        b_C_r(n, r, m, rp, t) = 0.0;
                        b_C_i(n, r, m, rp, t) = 0.0;
                     }

   for (int rp=0; rp<B1Nrows; rp++)
      for (int m=0; m<NsrcHex; m++)
         for (int r=0; r<B1Nrows; r++)
            for (int n=0; n<NsnkHex; n++)
               for (int t=0; t<Lt; t++) 
                  for (int x_out=0; x_out<Vsnk/sites_per_rank; x_out++)
                     for (int x_in = 0; x_in < sites_per_rank; ++x_in)
                     {
                        b_C_r_full(n, r, m, rp, x_in, x_out, t) = 0.0;
                        b_C_i_full(n, r, m, rp, x_in, x_out, t) = 0.0;
                     }

   if (rank == 0) {
   printf("prop 1 %4.9f + I %4.9f \n", b_B1_prop_r(0,0,0,0,0,0,0), b_B1_prop_i(0,0,0,0,0,0,0));
   printf("psi src 1 %4.9f + I %4.9f \n", b_B1_src_psi_r(0,0), b_B1_src_psi_i(0,0));
   printf("psi snk %4.9f + I %4.9f \n", b_B1_snk_psi_r(0,0,0,0), b_B1_snk_psi_i(0,0,0,0));
   printf("weights snk %4.9f \n", b_snk_weights(0,0));
   }
   gpu_tiramisu_make_fused_identical_baryon_blocks_correlator(
				    b_C_r.raw_buffer(),
				    b_C_i.raw_buffer(),
				    b_C_r_full.raw_buffer(),
				    b_C_i_full.raw_buffer(),
				    b_B1_prop_r.raw_buffer(),
				    b_B1_prop_i.raw_buffer(),
                b_B1_src_psi_r.raw_buffer(),
                b_B1_src_psi_i.raw_buffer(),
                b_B1_snk_psi_r.raw_buffer(),
                b_B1_snk_psi_i.raw_buffer(),
				    b_src_color_weights.raw_buffer(),
				    b_src_spin_weights.raw_buffer(),
				    b_src_weights.raw_buffer(),
				    b_src_spins.raw_buffer(),
				    b_snk_color_weights.raw_buffer(),
				    b_snk_spin_weights.raw_buffer(),
				    b_snk_weights.raw_buffer(),
				    b_sigs.raw_buffer());

   for (int rp=0; rp<B1Nrows; rp++)
      for (int m=0; m<NsrcHex; m++)
         for (int r=0; r<B1Nrows; r++)
            for (int n=0; n<NsnkHex; n++)
               for (int t=0; t<Lt; t++)
               {
                  C_re[index_5d(rp,m,r,n,t, NsrcHex,B1Nrows,NsnkHex,Lt)] = b_C_r(n,r,m,rp,t);
                  C_im[index_5d(rp,m,r,n,t, NsrcHex,B1Nrows,NsnkHex,Lt)] = b_C_i(n,r,m,rp,t);
               }

    if (rank == 0) {
   for (int rp=0; rp<B1Nrows; rp++) {
      printf("\n");
      for (int m=0; m<NsrcHex; m++)
         for (int r=0; r<B1Nrows; r++)
            for (int n=0; n<NsnkHex; n++)
               for (int t=0; t<Lt; t++) {
                  printf("rp=%d, m=%d, r=%d, n=%d, t=%d: %4.9f + I (%4.9f) \n", rp, m, r, n, t, C_re[index_5d(rp,m,r,n,t, NsrcHex,B1Nrows,NsnkHex,Lt)],  C_im[index_5d(rp,m,r,n,t, NsrcHex,B1Nrows,NsnkHex,Lt)]);
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

   int q, t, iC, iS, jC, jS, y, x, x1, x2, m, n, k, wnum, rp, r;
   int iC1, iS1, iC2, iS2, jC1, jS1, jC2, jS2, kC1, kS1, kC2, kS2;

   // Initialization
   // Props
   double* B1_prop_re = (double *) malloc(Lt * Nc * Ns * Nc * Ns * Vsnk * Vsrc * sizeof (double));
   double* B1_prop_im = (double *) malloc(Lt * Nc * Ns * Nc * Ns * Vsnk * Vsrc * sizeof (double));
   double* full_B1_prop_re = (double *) malloc(Lt * Nc * Ns * Nc * Ns * Vsnk * Vsrc * Nq * sizeof (double));
   double* full_B1_prop_im = (double *) malloc(Lt * Nc * Ns * Nc * Ns * Vsnk * Vsrc * Nq * sizeof (double));
      for (t = 0; t < Lt; t++) {
         for (iC = 0; iC < Nc; iC++) {
            for (iS = 0; iS < Ns; iS++) {
               for (jC = 0; jC < Nc; jC++) {
                  for (jS = 0; jS < Ns; jS++) {
                     for (y = 0; y < Vsrc; y++) {
                        for (x = 0; x < Vsnk; x++) {
	                   double v1 = rand()%10;
	                   double v2 = rand()%10;
	                   double v3 = rand()%10;
	                   double v4 = rand()%10;
			   if (randommode == 1) {
                              B1_prop_re[id_prop_index(t,jC,jS,iC,iS,y,x ,Nc,Ns,Vsrc,Vsnk,Lt)] = v1;
                              B1_prop_im[id_prop_index(t,jC,jS,iC,iS,y,x ,Nc,Ns,Vsrc,Vsnk,Lt)] = v3;
			   }
			   else {
                           if ((jC == iC) && (jS == iS)) {
                              B1_prop_re[id_prop_index(t,jC,jS,iC,iS,y,x ,Nc,Ns,Vsrc,Vsnk,Lt)] = 1/mq*cos(2*M_PI/6);
                              B1_prop_im[id_prop_index(t,jC,jS,iC,iS,y,x ,Nc,Ns,Vsrc,Vsnk,Lt)] = 1/mq*sin(2*M_PI/6);
                           }
                           else {
                              B1_prop_re[id_prop_index(t,jC,jS,iC,iS,y,x ,Nc,Ns,Vsrc,Vsnk,Lt)] = 0;
                              B1_prop_im[id_prop_index(t,jC,jS,iC,iS,y,x ,Nc,Ns,Vsrc,Vsnk,Lt)] = 0;
                           }
			   }
                           for (q = 0; q < Nq; q++) {
			   if (randommode == 1) {
                              full_B1_prop_re[prop_index(q,t,jC,jS,iC,iS,y,x ,Nc,Ns,Vsrc,Vsnk,Lt)] = v1;
                              full_B1_prop_im[prop_index(q,t,jC,jS,iC,iS,y,x ,Nc,Ns,Vsrc,Vsnk,Lt)] = v3;
			   }
			   else {
                           if ((jC == iC) && (jS == iS)) {
                              full_B1_prop_re[prop_index(q,t,jC,jS,iC,iS,y,x ,Nc,Ns,Vsrc,Vsnk,Lt)] = 1/mq*cos(2*M_PI/6);
                              full_B1_prop_im[prop_index(q,t,jC,jS,iC,iS,y,x ,Nc,Ns,Vsrc,Vsnk,Lt)] = 1/mq*sin(2*M_PI/6);
                           }
                           else {
                              full_B1_prop_re[prop_index(q,t,jC,jS,iC,iS,y,x ,Nc,Ns,Vsrc,Vsnk,Lt)] = 0;
                              full_B1_prop_im[prop_index(q,t,jC,jS,iC,iS,y,x ,Nc,Ns,Vsrc,Vsnk,Lt)] = 0;
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
   for (m = 0; m < Nsrc; m++)
      for (x = 0; x < Vsrc; x++) {
	      double v1 = 1.0;
	      double v2 = 0.0;
	      if (randommode == 1) {
	      v1 = rand()%10;
	      v2 = rand()%10;
	      }
         src_psi_B1_re[index_2d(x,m ,Nsrc)] = v1 ;// / Vsrc;
         src_psi_B1_im[index_2d(x,m ,Nsrc)] = v2 ;// / Vsrc;
      }
   double* snk_psi_B1_re = (double *) malloc(Nsnk * Vsnk * sizeof (double));
   double* snk_psi_B1_im = (double *) malloc(Nsnk * Vsnk * sizeof (double));
   for (n = 0; n < Nsnk; n++) {
      for (x = 0; x < Vsnk; x++) {
	      double v1 = 1.0;
	      double v2 = 0.0;
	      if (randommode == 1) {
	      v1 = rand()%10;
	      v2 = rand()%10;
	      }
         snk_psi_B1_re[index_2d(x,n ,Nsnk)] = v1  ;// / Vsnk;
         snk_psi_B1_im[index_2d(x,n ,Nsnk)] = v2 ;// / Vsnk;
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
   static double src_weights_r2_P[Nw] = {-1/ sqrt(2), 1/sqrt(2), 1/sqrt(2), -1/sqrt(2), 1/sqrt(2), -1/sqrt(2), 1/ sqrt(2), -1/sqrt(2), -1/sqrt(2), 1/sqrt(2), -1/sqrt(2), 1/sqrt(2)};

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
   int perms_array[2][3] = { {1,2,3}, {3,2,1} };
   int sigs_array[2] = {1,-1};
   int* perms = (int *) malloc(B1Nperms * Nq * sizeof (int));
   int sigs[B1Nperms];
   int permnum = 0;
   for (int i = 0; i < B1Nperms; i++) {
         for (int q = 0; q < Nq; q++) {
            perms[index_2d(permnum,q ,Nq)] = perms_array[i][q];
         }
         sigs[permnum] = sigs_array[i];
         permnum += 1;
   }
   // Correlators
   double* C_re = (double *) malloc(B1Nrows * B1Nrows * (NsrcHex) * (NsnkHex) * Lt * sizeof (double));
   double* C_im = (double *) malloc(B1Nrows * B1Nrows * (NsrcHex) * (NsnkHex) * Lt * sizeof (double));
   double* t_C_re = (double *) malloc(B1Nrows * B1Nrows * (NsrcHex) * (NsnkHex) * Lt * sizeof (double));
   double* t_C_im = (double *) malloc(B1Nrows * B1Nrows * (NsrcHex) * (NsnkHex) * Lt * sizeof (double));
   for (rp=0; rp<B1Nrows; rp++)
      for (m=0; m<NsrcHex; m++)
         for (r=0; r<B1Nrows; r++)
            for (n=0; n<NsnkHex; n++)
               for (t=0; t<Lt; t++) {
                  C_re[index_5d(rp,m,r,n,t, NsrcHex,B1Nrows,NsnkHex,Lt)] = 0.0;
                  C_im[index_5d(rp,m,r,n,t, NsrcHex,B1Nrows,NsnkHex,Lt)] = 0.0;
                  t_C_re[index_5d(rp,m,r,n,t, NsrcHex,B1Nrows,NsnkHex,Lt)] = 0.0;
                  t_C_im[index_5d(rp,m,r,n,t, NsrcHex,B1Nrows,NsnkHex,Lt)] = 0.0;
               }

   if (rank == 0)
   std::cout << "Start Tiramisu code." <<  std::endl;

   for (int i = 0; i < nb_tests; i++)
   {
      if (rank == 0)
         std::cout << "Run " << i << "/" << nb_tests <<  std::endl;
      auto start1 = std::chrono::high_resolution_clock::now();
       gpu_tiramisu_make_identical_nucleon_2pt(t_C_re,
           t_C_im,
           B1_prop_re, 
           B1_prop_im, 
           src_color_weights_r1,
           src_spin_weights_r1,
           src_weights_r1,
           src_color_weights_r2,
           src_spin_weights_r2,
           src_weights_r2,
           perms,
           sigs,
           src_psi_B1_re, 
           src_psi_B1_im, 
           snk_psi_B1_re,
           snk_psi_B1_im); //, Nc, Ns, Vsrc, Vsnk, Lt, Nw, Nq, NsrcHex, NsnkHex); 
       
      auto end1 = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double,std::milli> duration1 = end1 - start1;
      duration_vector_1.push_back(duration1);
   }

   if (rank == 0) {
    std::cout << "End Tiramisu code." <<  std::endl;
    std::cout << "Tiramisu execution time: " <<  (median(duration_vector_1)/1000.) << std::endl;

   for (rp=0; rp<B1Nrows; rp++) {
      printf("\n");
      for (m=0; m<NsrcHex; m++)
         for (r=0; r<B1Nrows; r++)
            for (n=0; n<NsnkHex; n++)
               for (t=0; t<Lt; t++) {
                  printf("rp=%d, m=%d, r=%d, n=%d, t=%d: %4.9f + I (%4.9f) \n", rp, m, r, n, t, t_C_re[index_5d(rp,m,r,n,t, NsrcHex,B1Nrows,NsnkHex,Lt)],  t_C_im[index_5d(rp,m,r,n,t, NsrcHex,B1Nrows,NsnkHex,Lt)]);
            }
   }



#if RUN_REFERENCE
   std::cout << "Start reference C code." <<  std::endl;
   for (int i = 0; i < nb_tests; i++)
   {
	   std::cout << "Run " << i << "/" << nb_tests <<  std::endl;
	   auto start2 = std::chrono::high_resolution_clock::now();

      make_nucleon_2pt(C_re, C_im, full_B1_prop_re, full_B1_prop_im, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, src_psi_B1_re, src_psi_B1_im, snk_psi_B1_re, snk_psi_B1_im, Nc, Ns, Vsrc, Vsnk, Lt, Nw, Nq, NsrcHex, NsnkHex);

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
    
   for (rp=0; rp<B1Nrows; rp++) {
      printf("\n");
      for (m=0; m<NsrcHex; m++)
         for (r=0; r<B1Nrows; r++)
            for (n=0; n<NsnkHex; n++)
               for (t=0; t<Lt; t++) {
                 double diff = std::sqrt(std::pow(std::abs(C_re[index_5d(rp,m,r,n,t, NsrcHex,B1Nrows,NsnkHex,Lt)] - t_C_re[index_5d(rp,m,r,n,t, NsrcHex,B1Nrows,NsnkHex,Lt)]),2) + std::pow(std::abs(C_im[index_5d(rp,m,r,n,t, NsrcHex,B1Nrows,NsnkHex,Lt)] -  t_C_im[index_5d(rp,m,r,n,t, NsrcHex,B1Nrows,NsnkHex,Lt)]),2));
                 double mag = std::sqrt(std::pow(std::abs(C_re[index_5d(rp,m,r,n,t, NsrcHex,B1Nrows,NsnkHex,Lt)]),2) + std::pow(std::abs(C_im[index_5d(rp,m,r,n,t, NsrcHex,B1Nrows,NsnkHex,Lt)]),2));
                 if (diff / mag >= 1/1e12 )
	            {
                  printf("rp=%d, m=%d, n=%d, t=%d: %4.9f + I (%4.9f) vs %4.9f + I (%4.9f) \n", rp, m, n, t, C_re[index_5d(rp,m,r,n,t, NsrcHex,B1Nrows,NsnkHex,Lt)], C_im[index_5d(rp,m,r,n,t, NsrcHex,B1Nrows,NsnkHex,Lt)],  t_C_re[index_5d(rp,m,r,n,t, NsrcHex,B1Nrows,NsnkHex,Lt)],  t_C_im[index_5d(rp,m,r,n,t, NsrcHex,B1Nrows,NsnkHex,Lt)]);
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
#endif

    return 0;
}

#ifdef __cplusplus
}  // extern "C"
#endif
