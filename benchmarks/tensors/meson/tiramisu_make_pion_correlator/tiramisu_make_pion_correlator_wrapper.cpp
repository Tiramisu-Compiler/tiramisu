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

#include "tiramisu_make_pion_correlator_wrapper.h"
#include "tiramisu_make_pion_correlator_ref.cpp"

#define RUN_REFERENCE 1
#define RUN_CHECK 1
int nb_tests = 1;
int randommode = 1;



void tiramisu_make_pion_2pt(double* C_re,
    double* C_im,
     double* prop_re, 
     double* prop_im, 
     int *src_color_weights,
     int *src_spin_weights,
     double *src_weights,
     double* src_psi_re, 
     double* src_psi_im, 
     double* snk_psi_re, 
     double* snk_psi_im)
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
    std::cout << "Array sizes" << std::endl;
    std::cout << "Prop:" <<  std::endl;
    std::cout << "	Max index size = " << Mq*Vsnk*Vsrc*Nc*NsFull*Nc*NsFull*Lt <<  std::endl;
    std::cout << "	Array size = " << Mq*Vsnk*Vsrc*Nc*NsFull*Nc*NsFull*Lt*sizeof(std::complex<double>)/mega << " Mega bytes" << std::endl;
    }

   // Halide buffers
   Halide::Buffer<double> b_C_r(NsnkHex, B0Nrows, NsrcHex, B0Nrows, Vsnk/sites_per_rank, Lt, "C_r");
   Halide::Buffer<double> b_C_i(NsnkHex, B0Nrows, NsrcHex, B0Nrows, Vsnk/sites_per_rank, Lt, "C_i");

   Halide::Buffer<int> b_src_color_weights(Mq, Mw, B0Nrows, "src_color_weights");
   Halide::Buffer<int> b_src_spin_weights(Mq, Mw, B0Nrows, "src_spin_weights");
   Halide::Buffer<double> b_src_weights(Mw, B0Nrows, "src_weights");

   Halide::Buffer<int> b_snk_color_weights(Mq, Mw, B0Nrows, "snk_color_weights");
   Halide::Buffer<int> b_snk_spin_weights(Mq, Mw, B0Nrows, "snk_spin_weights");
   Halide::Buffer<double> b_snk_weights(Mw, B0Nrows, "snk_weights");

    // prop
    Halide::Buffer<double> b_prop_r((double *)prop_re, {Vsrc, Vsnk, NsFull, Nc, NsFull, Nc, Lt, Mq});
    Halide::Buffer<double> b_prop_i((double *)prop_im, {Vsrc, Vsnk, NsFull, Nc, NsFull, Nc, Lt, Mq});

    if (rank == 0)
   printf("prop elem %4.9f \n", b_prop_r(0,0,0,0,0,0,0,0));

    // psi
    Halide::Buffer<double> b_src_psi_r((double *)src_psi_re, {NsrcHex, Vsrc});
    Halide::Buffer<double> b_src_psi_i((double *)src_psi_im, {NsrcHex, Vsrc});
    Halide::Buffer<double> b_snk_psi_r((double *)snk_psi_re, {NsnkHex, sites_per_rank, Vsnk/sites_per_rank});
    Halide::Buffer<double> b_snk_psi_i((double *)snk_psi_im, {NsnkHex, sites_per_rank, Vsnk/sites_per_rank});

   // Weights
 
   int* snk_color_weights = (int *) malloc(Mw * Mq * sizeof (int));
   int* snk_spin_weights = (int *) malloc(Mw * Mq * sizeof (int));
   for (int nB1=0; nB1<Mw; nB1++) {
         b_src_weights(nB1, 0) = src_weights[nB1];
         b_snk_weights(nB1, 0) = src_weights[nB1];
         for (int nq=0; nq<Mq; nq++) {
            snk_color_weights[index_2d(nB1,nq ,Mq)] = src_color_weights[index_2d(nB1,nq ,Mq)];
            snk_spin_weights[index_2d(nB1,nq ,Mq)] = src_spin_weights[index_2d(nB1,nq ,Mq)];
         }
   }
      for (int wnum=0; wnum< Mw; wnum++) {
         b_src_color_weights(0, wnum, 0) = snk_color_weights[index_2d(wnum,0 ,Mq)];
         b_src_spin_weights(0, wnum, 0) = snk_spin_weights[index_2d(wnum,0 ,Mq)];
         b_src_color_weights(1, wnum, 0) = snk_color_weights[index_2d(wnum,1 ,Mq)];
         b_src_spin_weights(1, wnum, 0) = snk_spin_weights[index_2d(wnum,1 ,Mq)];

         b_snk_color_weights(0, wnum, 0) = snk_color_weights[index_2d(wnum,0 ,Mq)];
         b_snk_spin_weights(0, wnum, 0) = snk_spin_weights[index_2d(wnum,0 ,Mq)];
         b_snk_color_weights(1, wnum, 0) = snk_color_weights[index_2d(wnum,1 ,Mq)];
         b_snk_spin_weights(1, wnum, 0) = snk_spin_weights[index_2d(wnum,1 ,Mq)];
      }

   for (int rp=0; rp<B0Nrows; rp++)
      for (int m=0; m<NsrcHex; m++)
         for (int r=0; r<B0Nrows; r++)
            for (int n=0; n<NsnkHex; n++)
               for (int t=0; t<Lt; t++) 
                  for (int x=0; x<Vsnk/sites_per_rank; x++) {
                     b_C_r(n,r,m,rp,x,t) = 0.0;
                     b_C_i(n,r,m,rp,x,t) = 0.0;
                  } 

   if (rank == 0) {
   printf("prop 1 %4.9f + I %4.9f \n", b_prop_r(0,0,0,0,0,0,0,0), b_prop_i(0,0,0,0,0,0,0,0));
   printf("psi src 1 %4.9f + I %4.9f \n", b_src_psi_r(0,0), b_src_psi_i(0,0));
   printf("psi snk %4.9f + I %4.9f \n", b_snk_psi_r(0,0,0,0), b_snk_psi_i(0,0,0,0));
   printf("weights snk %4.9f \n", b_snk_weights(0,0));
   }
   tiramisu_make_pion_correlator(
				    b_C_r.raw_buffer(),
				    b_C_i.raw_buffer(),
				    b_prop_r.raw_buffer(),
				    b_prop_i.raw_buffer(),
                b_src_psi_r.raw_buffer(),
                b_src_psi_i.raw_buffer(),
                b_snk_psi_r.raw_buffer(),
                b_snk_psi_i.raw_buffer(),
				    b_src_color_weights.raw_buffer(),
				    b_src_spin_weights.raw_buffer(),
				    b_src_weights.raw_buffer(),
				    b_snk_color_weights.raw_buffer(),
				    b_snk_spin_weights.raw_buffer(),
				    b_snk_weights.raw_buffer()); 

   if (rank == 0) {
   printf("non-zero r1? %4.1f + I %4.1f ", b_C_r(0,0,0,0,0,0), b_C_i(0,0,0,0,0,0) );
   }

    // symmetrize and such
#ifdef WITH_MPI
   
   for (int rp=0; rp<B0Nrows; rp++)
      for (int m=0; m<NsrcHex; m++)
         for (int r=0; r<B0Nrows; r++)
            for (int n=0; n<NsnkHex; n++)
               for (int t=0; t<Lt; t++)  {
                  double number0r;
                  double number0i;
                  double this_number0r = b_C_r(n,r,m,rp,rank,t);
                  double this_number0i = b_C_i(n,r,m,rp,rank,t);
                  MPI_Allreduce(&this_number0r, &number0r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                  MPI_Allreduce(&this_number0i, &number0i, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                  C_re[index_5d(rp,m,r,n,t, NsrcHex,B0Nrows,NsnkHex,Lt)] += number0r;
                  C_im[index_5d(rp,m,r,n,t, NsrcHex,B0Nrows,NsnkHex,Lt)] += number0i;
               }
#else
   for (int rp=0; rp<B0Nrows; rp++)
      for (int m=0; m<NsrcHex; m++)
         for (int r=0; r<B0Nrows; r++)
            for (int n=0; n<NsnkHex; n++)
               for (int t=0; t<Lt; t++)
                for (int x=0; x<Vsnk; x++) {
                  double number0r;
                  double number0i;
                  number0r = b_C_r(n,r,m,rp,x,t);
                  number0i = b_C_i(n,r,m,rp,x,t);
                  C_re[index_5d(rp,m,r,n,t, NsrcHex,B0Nrows,NsnkHex,Lt)] += number0r;
                  C_im[index_5d(rp,m,r,n,t, NsrcHex,B0Nrows,NsnkHex,Lt)] += number0i;
               }
#endif

    if (rank == 0) {
   for (int rp=0; rp<B0Nrows; rp++) {
      printf("\n");
      for (int m=0; m<NsrcHex; m++)
         for (int r=0; r<B0Nrows; r++)
            for (int n=0; n<NsnkHex; n++)
               for (int t=0; t<Lt; t++) {
                  printf("rp=%d, m=%d, r=%d, n=%d, t=%d: %4.1f + I (%4.1f) \n", rp, m, r, n, t, C_re[index_5d(rp,m,r,n,t, NsrcHex,B0Nrows,NsnkHex,Lt)],  C_im[index_5d(rp,m,r,n,t, NsrcHex,B0Nrows,NsnkHex,Lt)]);
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
   double* prop_re = (double *) malloc(Mq * Lt * Nc * NsFull * Nc * NsFull * Vsnk * Vsrc * sizeof (double));
   double* prop_im = (double *) malloc(Mq * Lt * Nc * NsFull * Nc * NsFull * Vsnk * Vsrc * sizeof (double));
   for (q = 0; q < Mq; q++) {
      for (t = 0; t < Lt; t++) {
         for (iC = 0; iC < Nc; iC++) {
            for (iS = 0; iS < NsFull; iS++) {
               for (jC = 0; jC < Nc; jC++) {
                  for (jS = 0; jS < NsFull; jS++) {
                     for (y = 0; y < Vsrc; y++) {
                        for (x = 0; x < Vsnk; x++) {
			   if (randommode == 1) {
	                        double v1 = rand()%10;
	                        double v2 = rand()%10;
	                        double v3 = rand()%10;
	                        double v4 = rand()%10;
                           prop_re[prop_index(q,t,jC,jS,iC,iS,y,x ,Nc,NsFull,Vsrc,Vsnk,Lt)] = v1;
                           if (q == 0)
                               prop_im[prop_index(q,t,jC,jS,iC,iS,y,x ,Nc,NsFull,Vsrc,Vsnk,Lt)] = v3;
                           else
                               prop_im[prop_index(q,t,jC,jS,iC,iS,y,x ,Nc,NsFull,Vsrc,Vsnk,Lt)] = -v3;
			   }
			   else {
                           if ((jC == iC) && (jS == iS)) {
                              prop_re[prop_index(q,t,jC,jS,iC,iS,y,x ,Nc,NsFull,Vsrc,Vsnk,Lt)] = 1/mq*cos(2*M_PI/6);
                              if (q == 0)
                                  prop_im[prop_index(q,t,jC,jS,iC,iS,y,x ,Nc,NsFull,Vsrc,Vsnk,Lt)] = 1/mq*sin(2*M_PI/6);
                              else
                                  prop_im[prop_index(q,t,jC,jS,iC,iS,y,x ,Nc,NsFull,Vsrc,Vsnk,Lt)] = -1/mq*sin(2*M_PI/6);
                           }
                           else {
                              prop_re[prop_index(q,t,jC,jS,iC,iS,y,x ,Nc,NsFull,Vsrc,Vsnk,Lt)] = 0;
                              prop_im[prop_index(q,t,jC,jS,iC,iS,y,x ,Nc,NsFull,Vsrc,Vsnk,Lt)] = 0;
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
   double* src_psi_re = (double *) malloc(Nsrc * Vsrc * sizeof (double));
   double* src_psi_im = (double *) malloc(Nsrc * Vsrc * sizeof (double));
   for (m = 0; m < Nsrc; m++)
      for (x = 0; x < Vsrc; x++) {
	      double v1 = 1.0;
	      double v2 = 0.0;
	      if (randommode == 1) {
	      v1 = rand()%10;
	      v2 = rand()%10;
	      }
         src_psi_re[index_2d(x,m ,Nsrc)] = v1 ;// / Vsrc;
         src_psi_im[index_2d(x,m ,Nsrc)] = v2 ;// / Vsrc;
      }
   double* snk_psi_re = (double *) malloc(Nsnk * Vsnk * sizeof (double));
   double* snk_psi_im = (double *) malloc(Nsnk * Vsnk * sizeof (double));
   for (n = 0; n < Nsnk; n++) {
      for (x = 0; x < Vsnk; x++) {
	      double v1 = 1.0;
	      double v2 = 0.0;
	      if (randommode == 1) {
	      v1 = rand()%10;
	      v2 = rand()%10;
	      }
         snk_psi_re[index_2d(x,n ,Nsnk)] = v1  ;// / Vsnk;
         snk_psi_im[index_2d(x,n ,Nsnk)] = v2 ;// / Vsnk;
      }
   }
   // Weights
   /*static int src_color_weights_r1_P[Mw][Mq] = { {0,1,2}, {0,2,1}, {1,0,2} ,{0,1,2}, {0,2,1}, {1,0,2}, {1,2,0}, {2,1,0}, {2,0,1} };
   static int src_spin_weights_r1_P[Mw][Mq] = { {0,1,0}, {0,1,0}, {0,1,0}, {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0} };
   static double src_weights_r1_P[Mw] = {-2/ sqrt(2), 2/sqrt(2), 2/sqrt(2), 1/sqrt(2), -1/sqrt(2), -1/sqrt(2), 1/sqrt(2), -1/sqrt(2), 1/sqrt(2)};
   static int src_color_weights_r2_P[Mw][Mq] = { {0,1,2}, {0,2,1}, {1,0,2} ,{1,2,0}, {2,1,0}, {2,0,1}, {0,1,2}, {0,2,1}, {1,0,2} };
   static int src_spin_weights_r2_P[Mw][Mq] = { {0,1,1}, {0,1,1}, {0,1,1}, {0,1,1}, {0,1,1}, {0,1,1}, {1,0,1}, {1,0,1}, {1,0,1} };
   static double src_weights_r2_P[Mw] = {1/ sqrt(2), -1/sqrt(2), -1/sqrt(2), 1/sqrt(2), -1/sqrt(2), 1/sqrt(2), -2/sqrt(2), 2/sqrt(2), 2/sqrt(2)}; */
   
   static int src_color_weights_P[Mw][Mq] = { {0,0}, {1,1}, {2,2}, {0,0}, {1,1}, {2,2}, {0,0}, {1,1}, {2,2}, {0,0}, {1,1}, {2,2}};
   static int src_spin_weights_P[Mw][Mq] =  { {0,0}, {0,0}, {0,0}, {1,1}, {1,1}, {1,1}, {2,2}, {2,2}, {2,2}, {3,3}, {3,3}, {3,3}};
   static double src_weights_P[Mw] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0};
   //static double src_weights_P[Mw] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

   int* src_color_weights = (int *) malloc(Mw * Mq * sizeof (int));
   int* src_spin_weights = (int *) malloc(Mw * Mq * sizeof (int));
   double src_weights[Mw];
   for (wnum = 0; wnum < Mw; wnum++) {
      for (q = 0; q < Mq; q++) {
         src_color_weights[index_2d(wnum,q ,Mq)] = src_color_weights_P[wnum][q];
         src_spin_weights[index_2d(wnum,q ,Mq)] = src_spin_weights_P[wnum][q];
      }
      src_weights[wnum] = src_weights_P[wnum];
   }
   // Correlators
   double* C_re = (double *) malloc(B0Nrows * B0Nrows * (NsrcHex) * (NsnkHex) * Lt * sizeof (double));
   double* C_im = (double *) malloc(B0Nrows * B0Nrows * (NsrcHex) * (NsnkHex) * Lt * sizeof (double));
   double* t_C_re = (double *) malloc(B0Nrows * B0Nrows * (NsrcHex) * (NsnkHex) * Lt * sizeof (double));
   double* t_C_im = (double *) malloc(B0Nrows * B0Nrows * (NsrcHex) * (NsnkHex) * Lt * sizeof (double));
   for (rp=0; rp<B0Nrows; rp++)
      for (m=0; m<NsrcHex; m++)
         for (r=0; r<B0Nrows; r++)
            for (n=0; n<NsnkHex; n++)
               for (t=0; t<Lt; t++) {
                  C_re[index_5d(rp,m,r,n,t, NsrcHex,B0Nrows,NsnkHex,Lt)] = 0.0;
                  C_im[index_5d(rp,m,r,n,t, NsrcHex,B0Nrows,NsnkHex,Lt)] = 0.0;
                  t_C_re[index_5d(rp,m,r,n,t, NsrcHex,B0Nrows,NsnkHex,Lt)] = 0.0;
                  t_C_im[index_5d(rp,m,r,n,t, NsrcHex,B0Nrows,NsnkHex,Lt)] = 0.0;
               }

   if (rank == 0)
   std::cout << "Start Tiramisu code." <<  std::endl;

   for (int i = 0; i < nb_tests; i++)
   {
      if (rank == 0)
         std::cout << "Run " << i << "/" << nb_tests <<  std::endl;
      auto start1 = std::chrono::high_resolution_clock::now();

       tiramisu_make_pion_2pt(t_C_re,
           t_C_im,
           prop_re, 
           prop_im, 
           src_color_weights,
           src_spin_weights,
           src_weights,
           src_psi_re, 
           src_psi_im, 
           snk_psi_re,
           snk_psi_im); //, Nc, Ns, Vsrc, Vsnk, Lt, Mw, Mq, NsrcHex, NsnkHex);
       
      auto end1 = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double,std::milli> duration1 = end1 - start1;
      duration_vector_1.push_back(duration1);
   }

   if (rank == 0) {
    std::cout << "End Tiramisu code." <<  std::endl;

   for (rp=0; rp<B0Nrows; rp++) {
      printf("\n");
      for (m=0; m<NsrcHex; m++)
         for (r=0; r<B0Nrows; r++)
            for (n=0; n<NsnkHex; n++)
               for (t=0; t<Lt; t++) {
                  printf("rp=%d, m=%d, r=%d, n=%d, t=%d: %4.1f + I (%4.1f) \n", rp, m, r, n, t, t_C_re[index_5d(rp,m,r,n,t, NsrcHex,B0Nrows,NsnkHex,Lt)],  t_C_im[index_5d(rp,m,r,n,t, NsrcHex,B0Nrows,NsnkHex,Lt)]);
            }
   }



#if RUN_REFERENCE
   std::cout << "Start reference C code." <<  std::endl;
   for (int i = 0; i < nb_tests; i++)
   {
	   std::cout << "Run " << i << "/" << nb_tests <<  std::endl;
	   auto start2 = std::chrono::high_resolution_clock::now();

      make_pion_2pt(C_re, C_im, prop_re, prop_im, src_color_weights, src_spin_weights, src_weights, src_color_weights, src_spin_weights, src_weights, src_psi_re, src_psi_im, snk_psi_re, snk_psi_im, Nc, NsFull, Vsrc, Vsnk, Lt, Mw, Mq, NsrcHex, NsnkHex);

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
    
   for (rp=0; rp<B0Nrows; rp++) {
      printf("\n");
      for (m=0; m<NsrcHex; m++)
         for (r=0; r<B0Nrows; r++)
            for (n=0; n<NsnkHex; n++)
               for (t=0; t<Lt; t++) {
                  if ((std::abs(C_re[index_5d(rp,m,r,n,t, NsrcHex,B0Nrows,NsnkHex,Lt)] - t_C_re[index_5d(rp,m,r,n,t, NsrcHex,B0Nrows,NsnkHex,Lt)]) >= 0.01*Vsnk*Vsnk) ||
	               (std::abs(C_im[index_5d(rp,m,r,n,t, NsrcHex,B0Nrows,NsnkHex,Lt)] -  t_C_im[index_5d(rp,m,r,n,t, NsrcHex,B0Nrows,NsnkHex,Lt)]) >= 0.01*Vsnk*Vsnk))
	            {
                  printf("rp=%d, m=%d, n=%d, t=%d: %4.1f + I (%4.1f) vs %4.1f + I (%4.1f) \n", rp, m, n, t, C_re[index_5d(rp,m,r,n,t, NsrcHex,B0Nrows,NsnkHex,Lt)], C_im[index_5d(rp,m,r,n,t, NsrcHex,B0Nrows,NsnkHex,Lt)],  t_C_re[index_5d(rp,m,r,n,t, NsrcHex,B0Nrows,NsnkHex,Lt)],  t_C_im[index_5d(rp,m,r,n,t, NsrcHex,B0Nrows,NsnkHex,Lt)]);
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
