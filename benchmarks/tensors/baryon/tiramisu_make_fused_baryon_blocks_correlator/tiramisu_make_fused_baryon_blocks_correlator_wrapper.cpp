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

#include "tiramisu_make_fused_baryon_blocks_correlator_wrapper.h"
#include "tiramisu_make_fused_baryon_blocks_correlator_ref.cpp"

#define RUN_REFERENCE 1
#define RUN_CHECK 1
int nb_tests = 1;
int randommode = 1;



void tiramisu_make_nucleon_2pt(double* C_re,
    double* C_im,
     double* B1_prop_re, 
     double* B1_prop_im, 
     int *src_color_weights_r1,
     int *src_spin_weights_r1,
     double *src_weights_r1,
     int *src_color_weights_r2,
     int *src_spin_weights_r2,
     double *src_weights_r2,
     double* src_psi_B1_re, 
     double* src_psi_B1_im, 
     double* snk_psi_B1_re, 
     double* snk_psi_B1_im)
{

   int q, t, iC, iS, jC, jS, y, x, x1, x2, m, n, k, wnum, b;
   int iC1, iS1, iC2, iS2, jC1, jS1, jC2, jS2, kC1, kS1, kC2, kS2;

    int rank = 0;
#ifdef WITH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

  // printf("hi I'm rank %d \n", rank);

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
    std::cout << "	Max index size = " << Vsnk*NsrcHex*Nc*Ns*Nc*Ns*Nc*Ns <<  std::endl;
    std::cout << "	Array size = " << Vsnk*NsrcHex*Nc*Ns*Nc*Ns*Nc*Ns*sizeof(std::complex<double>)/kilo << " kilo bytes" <<  std::endl;
    std::cout << "Blocal, Bsingle, Bdouble:" <<  std::endl;
    std::cout << "	Max index size = " << Nc*Ns*Nc*Ns*Nc*Ns <<  std::endl;
    std::cout << "	Array size = " << Nc*Ns*Nc*Ns*Nc*Ns*sizeof(std::complex<double>)/kilo << " kilo bytes" <<  std::endl;
    std::cout << std::endl;
    }

   // Halide buffers
   Halide::Buffer<double> b_C_r(NsnkHex, B1Nrows, NsrcHex, Vsnk, Lt, "C_r");
   Halide::Buffer<double> b_C_i(NsnkHex, B1Nrows, NsrcHex, Vsnk, Lt, "C_i");

   Halide::Buffer<int> b_src_color_weights(Nq, Nw, B1Nrows, "src_color_weights");
   Halide::Buffer<int> b_src_spin_weights(Nq, Nw, B1Nrows, "src_spin_weights");
   Halide::Buffer<double> b_src_weights(Nw, B1Nrows, "src_weights");

   Halide::Buffer<int> b_snk_blocks(B1Nrows, "snk_blocks");
   Halide::Buffer<int> b_snk_color_weights(Nq, Nw, B1Nrows, "snk_color_weights");
   Halide::Buffer<int> b_snk_spin_weights(Nq, Nw, B1Nrows, "snk_spin_weights");
   Halide::Buffer<double> b_snk_weights(Nw, B1Nrows, "snk_weights");

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

   for (int b=0; b<B1Nrows; b++)
      for (int m=0; m<NsrcHex; m++)
         for (int n=0; n<NsnkHex; n++)
            for (int t=0; t<Lt; t++) 
               for (int x=0; x<Vsnk; x++) {
                  b_C_r(n,b,m,x,t) = 0.0;
                  b_C_i(n,b,m,x,t) = 0.0;
            } 

   if (rank == 0) {
   printf("prop 1 %4.9f + I %4.9f \n", b_B1_prop_r(0,0,0,0,0,0,0,0), b_B1_prop_i(0,0,0,0,0,0,0,0));
   printf("psi src 1 %4.9f + I %4.9f \n", b_B1_src_psi_r(0,0), b_B1_src_psi_i(0,0));
   printf("psi snk %4.9f + I %4.9f \n", b_B1_snk_psi_r(0,0,0), b_B1_snk_psi_i(0,0,0));
   printf("weights snk %4.9f \n", b_snk_weights(0,0));
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

   if (rank == 0) {
   printf("non-zero r1? %4.1f + I %4.1f ", b_C_r(0,0,0,0,0), b_C_i(0,0,0,0,0) );
   printf("non-zero r2? %4.1f + I %4.1f ", b_C_r(0,1,0,0,0), b_C_i(0,1,0,0,0) );
   }

    // symmetrize and such
    /*  for (int m=0; m<NsrcHex; m++)
         for (int n=0; n<NsnkHex; n++)
            for (int t=0; t<Lt; t++) 
               for (int x=0; x<Vsnk; x++) {
                  C_re[index_4d(0,m,n,t, NsrcHex,NsnkHex,Lt)] += b_C_r(n,0,m,t);
                  C_re[index_4d(1,m,n,t, NsrcHex,NsnkHex,Lt)] += b_C_r(n,1,m,t);
                  C_im[index_4d(0,m,n,t, NsrcHex,NsnkHex,Lt)] += b_C_i(n,0,m,t);
                  C_im[index_4d(1,m,n,t, NsrcHex,NsnkHex,Lt)] += b_C_i(n,1,m,t);
               } */
#ifdef WITH_MPI
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
                  C_re[index_4d(0,m,n,t, NsrcHex,NsnkHex,Lt)] += number0r;
                  C_re[index_4d(1,m,n,t, NsrcHex,NsnkHex,Lt)] += number1r;
                  C_im[index_4d(0,m,n,t, NsrcHex,NsnkHex,Lt)] += number0i;
                  C_im[index_4d(1,m,n,t, NsrcHex,NsnkHex,Lt)] += number1i; 
               }
#else
      for (int m=0; m<NsrcHex; m++)
         for (int n=0; n<NsnkHex; n++)
            for (int t=0; t<Lt; t++)  {
             for (int x=0; x<Vsnk; x++) {
               double number0r;
               double number0i;
               double number1r;
               double number1i; 
               number0r = b_C_r(n,0,m,x,t);
               number0i = b_C_i(n,0,m,x,t);
               number1r = b_C_r(n,1,m,x,t);
               number1i = b_C_i(n,1,m,x,t); 
                  C_re[index_4d(0,m,n,t, NsrcHex,NsnkHex,Lt)] += number0r;
                  C_re[index_4d(1,m,n,t, NsrcHex,NsnkHex,Lt)] += number1r;
                  C_im[index_4d(0,m,n,t, NsrcHex,NsnkHex,Lt)] += number0i;
                  C_im[index_4d(1,m,n,t, NsrcHex,NsnkHex,Lt)] += number1i; 
               }
#endif

    if (rank == 0) {
   for (int b=0; b<2; b++) {
      printf("\n");
      for (int m=0; m<NsrcHex; m++)
         for (int n=0; n<NsnkHex; n++)
            for (int t=0; t<Lt; t++) {
                  printf("r=%d, m=%d, n=%d, t=%d: %4.1f + I (%4.1f) \n", b, m, n, t, C_re[index_4d(b,m,n,t, NsrcHex,NsnkHex,Lt)],  C_im[index_4d(b,m,n,t, NsrcHex,NsnkHex,Lt)]);
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

   int q, t, iC, iS, jC, jS, y, x, x1, x2, m, n, k, wnum, b;
   int iC1, iS1, iC2, iS2, jC1, jS1, jC2, jS2, kC1, kS1, kC2, kS2;

   // Initialization
   // Props
   double* B1_prop_re = (double *) malloc(Nq * Lt * Nc * Ns * Nc * Ns * Vsnk * Vsrc * sizeof (double));
   double* B1_prop_im = (double *) malloc(Nq * Lt * Nc * Ns * Nc * Ns * Vsnk * Vsrc * sizeof (double));
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
			   }
			   else {
                           if ((jC == iC) && (jS == iS)) {
                              B1_prop_re[prop_index(q,t,jC,jS,iC,iS,y,x ,Nc,Ns,Vsrc,Vsnk,Lt)] = 1/mq*cos(2*M_PI/6);
                              B1_prop_im[prop_index(q,t,jC,jS,iC,iS,y,x ,Nc,Ns,Vsrc,Vsnk,Lt)] = 1/mq*sin(2*M_PI/6);
                           }
                           else {
                              B1_prop_re[prop_index(q,t,jC,jS,iC,iS,y,x ,Nc,Ns,Vsrc,Vsnk,Lt)] = 0;
                              B1_prop_im[prop_index(q,t,jC,jS,iC,iS,y,x ,Nc,Ns,Vsrc,Vsnk,Lt)] = 0;
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
   static int src_color_weights_r1_P[Nw][Nq] = { {0,1,2}, {0,2,1}, {1,0,2} ,{0,1,2}, {0,2,1}, {1,0,2}, {1,2,0}, {2,1,0}, {2,0,1} };
   static int src_spin_weights_r1_P[Nw][Nq] = { {0,1,0}, {0,1,0}, {0,1,0}, {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0} };
   static double src_weights_r1_P[Nw] = {-2/ sqrt(2), 2/sqrt(2), 2/sqrt(2), 1/sqrt(2), -1/sqrt(2), -1/sqrt(2), 1/sqrt(2), -1/sqrt(2), 1/sqrt(2)};

   static int src_color_weights_r2_P[Nw][Nq] = { {0,1,2}, {0,2,1}, {1,0,2} ,{1,2,0}, {2,1,0}, {2,0,1}, {0,1,2}, {0,2,1}, {1,0,2} };
   static int src_spin_weights_r2_P[Nw][Nq] = { {0,1,1}, {0,1,1}, {0,1,1}, {0,1,1}, {0,1,1}, {0,1,1}, {1,0,1}, {1,0,1}, {1,0,1} };
   static double src_weights_r2_P[Nw] = {1/ sqrt(2), -1/sqrt(2), -1/sqrt(2), 1/sqrt(2), -1/sqrt(2), 1/sqrt(2), -2/sqrt(2), 2/sqrt(2), 2/sqrt(2)};

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
   // Correlators
   double* C_re = (double *) malloc(2 * (NsrcHex) * (NsnkHex) * Lt * sizeof (double));
   double* C_im = (double *) malloc(2 * (NsrcHex) * (NsnkHex) * Lt * sizeof (double));
   double* t_C_re = (double *) malloc(2 * (NsrcHex) * (NsnkHex) * Lt * sizeof (double));
   double* t_C_im = (double *) malloc(2 * (NsrcHex) * (NsnkHex) * Lt * sizeof (double));
   for (b=0; b<2; b++)
      for (m=0; m<NsrcHex; m++)
         for (n=0; n<NsnkHex; n++)
            for (t=0; t<Lt; t++) {
               C_re[index_4d(b,m,n,t, NsrcHex,NsnkHex,Lt)] = 0.0;
               C_im[index_4d(b,m,n,t, NsrcHex,NsnkHex,Lt)] = 0.0;
               t_C_re[index_4d(b,m,n,t, NsrcHex,NsnkHex,Lt)] = 0.0;
               t_C_im[index_4d(b,m,n,t, NsrcHex,NsnkHex,Lt)] = 0.0;
            }

   if (rank == 0)
   std::cout << "Start Tiramisu code." <<  std::endl;

   for (int i = 0; i < nb_tests; i++)
   {
      if (rank == 0)
         std::cout << "Run " << i << "/" << nb_tests <<  std::endl;
      auto start1 = std::chrono::high_resolution_clock::now();

       tiramisu_make_nucleon_2pt(t_C_re,
           t_C_im,
           B1_prop_re, 
           B1_prop_im, 
           src_color_weights_r1,
           src_spin_weights_r1,
           src_weights_r1,
           src_color_weights_r2,
           src_spin_weights_r2,
           src_weights_r2,
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

   for (b=0; b<2; b++) {
      printf("\n");
      for (m=0; m<NsrcHex; m++)
         for (n=0; n<NsnkHex; n++)
            for (t=0; t<Lt; t++) {
                  printf("r=%d, m=%d, n=%d, t=%d: %4.1f + I (%4.1f) \n", b, m, n, t, t_C_re[index_4d(b,m,n,t, NsrcHex,NsnkHex,Lt)],  t_C_im[index_4d(b,m,n,t, NsrcHex,NsnkHex,Lt)]);
            }
   }



#if RUN_REFERENCE
   std::cout << "Start reference C code." <<  std::endl;
   for (int i = 0; i < nb_tests; i++)
   {
	   std::cout << "Run " << i << "/" << nb_tests <<  std::endl;
	   auto start2 = std::chrono::high_resolution_clock::now();

      make_nucleon_2pt(C_re, C_im, B1_prop_re, B1_prop_im, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, src_psi_B1_re, src_psi_B1_im, snk_psi_B1_re, snk_psi_B1_im, Nc, Ns, Vsrc, Vsnk, Lt, Nw, Nq, NsrcHex, NsnkHex);

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
    
   for (b=0; b<2; b++) {
      printf("\n");
      for (m=0; m<NsrcHex; m++)
         for (n=0; n<NsnkHex; n++)
            for (t=0; t<Lt; t++) {
               if ((std::abs(C_re[index_4d(b,m,n,t, NsrcHex,NsnkHex,Lt)] - t_C_re[index_4d(b,m,n,t, NsrcHex,NsnkHex,Lt)]) >= 0.01*Vsnk*Vsnk) ||
	               (std::abs(C_im[index_4d(b,m,n,t, NsrcHex,NsnkHex,Lt)] -  t_C_im[index_4d(b,m,n,t, NsrcHex,NsnkHex,Lt)]) >= 0.01*Vsnk*Vsnk))
	            {
                  printf("r=%d, m=%d, n=%d, t=%d: %4.1f + I (%4.1f) vs %4.1f + I (%4.1f) \n", b, m, n, t, C_re[index_4d(b,m,n,t, NsrcHex,NsnkHex,Lt)], C_im[index_4d(b,m,n,t, NsrcHex,NsnkHex,Lt)],  t_C_re[index_4d(b,m,n,t, NsrcHex,NsnkHex,Lt)],  t_C_im[index_4d(b,m,n,t, NsrcHex,NsnkHex,Lt)]);
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
