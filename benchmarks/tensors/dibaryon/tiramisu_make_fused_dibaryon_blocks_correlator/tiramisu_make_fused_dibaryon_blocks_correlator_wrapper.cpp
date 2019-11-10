#include "Halide.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>
#include "benchmarks.h"

#include "tiramisu_make_fused_dibaryon_blocks_correlator_wrapper.h"
#include "tiramisu_make_fused_dibaryon_blocks_correlator_ref.cpp"

#define RUN_REFERENCE 1
#define RUN_CHECK 1
int nb_tests = 3;

int main(int, char **)
{
   std::vector<std::chrono::duration<double,std::milli>> duration_vector_1;
   std::vector<std::chrono::duration<double,std::milli>> duration_vector_2;

   long kilo = 1024;

   std::cout << "Array sizes" << std::endl;
   std::cout << "Prop:" <<  std::endl;
   std::cout << "	Max index size = " << Nc*Ns*Nc*Ns*Lt*Vsrc*Vsnk*Nq <<  std::endl;
   std::cout << "	Array size = " << Nc*Ns*Nc*Ns*Lt*Vsrc*Vsnk*sizeof(std::complex<double>)/kilo << " kilobytes" << std::endl;
   std::cout << "Blocal:" <<  std::endl;
   std::cout << "	Max index size = " << Nsrc*Nc*Ns*Nc*Ns*Nc*Ns <<  std::endl;
   std::cout << "	Array size = " << Nsrc*Nc*Ns*Nc*Ns*Nc*Ns*sizeof(std::complex<double>) << " bytes" << std::endl;
   std::cout << "Bsingle, Bdouble, Q, O & P:" <<  std::endl;
   std::cout << "	Max index size = " << Nsrc*Nc*Ns*Nc*Ns*Nc*Ns <<  std::endl;
   std::cout << "	Array size = " << Nsrc*Nc*Ns*Nc*Ns*Nc*Ns*sizeof(std::complex<double>) << " bytes" <<  std::endl;
   std::cout << std::endl;

   int NsrcHex_f = 0;
   int NsnkHex_f = 0;
   int Nr = 4;
   int q, t, iC, iS, jC, jS, y, x, x1, x2, m, n, k, wnum, b;

   // Halide buffers
   //Halide::Buffer<double> b_C_r(Nr, Lt, Nsnk, Nsrc, "C_r");
   //Halide::Buffer<double> b_C_i(Nr, Lt, Nsnk, Nsrc, "C_i");
   Halide::Buffer<double> b_C_r(Lt, Nsnk, Nsrc, "C_r");
   Halide::Buffer<double> b_C_i(Lt, Nsnk, Nsrc, "C_i");

   Halide::Buffer<double> b_B1_prop_r(Vsrc, Vsnk, Ns, Nc, Ns, Nc, Lt, Nq, "B1_prop_r");
   Halide::Buffer<double> b_B1_prop_i(Vsrc, Vsnk, Ns, Nc, Ns, Nc, Lt, Nq, "B1_prop_i");
   Halide::Buffer<double> b_B2_prop_r(Vsrc, Vsnk, Ns, Nc, Ns, Nc, Lt, Nq, "B2_prop_r");
   Halide::Buffer<double> b_B2_prop_i(Vsrc, Vsnk, Ns, Nc, Ns, Nc, Lt, Nq, "B2_prop_i");
   Halide::Buffer<double> b_B1_src_psi_r(Vsrc, Nsrc, "B1_psi_r");
   Halide::Buffer<double> b_B1_src_psi_i(Vsrc, Nsrc, "B1_psi_i");
   Halide::Buffer<double> b_B2_src_psi_r(Vsrc, Nsrc, "B2_psi_r");
   Halide::Buffer<double> b_B2_src_psi_i(Vsrc, Nsrc, "B2_psi_i");

   Halide::Buffer<int> b_perms(2*Nq, Nperms, "perms");
   Halide::Buffer<int> b_sigs(Nperms, "sigs");
   Halide::Buffer<double> b_overall_weight(1, "overall_weight");
   Halide::Buffer<int> b_snk_color_weights(Nq, Nw2, 2, "snk_color_weights");
   Halide::Buffer<int> b_snk_spin_weights(Nq, Nw2, 2, "snk_spin_weights");
   Halide::Buffer<double> b_snk_weights(Nw2, "snk_weights");
   Halide::Buffer<double> b_snk_psi_r(Nsnk, Vsnk, Vsnk, "snk_psi_re");
   Halide::Buffer<double> b_snk_psi_i(Nsnk, Vsnk, Vsnk, "snk_psi_im");

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
                           if ((jC == iC) && (jS == iS)) {
                              B1_prop_re[prop_index(q,t,jC,jS,iC,iS,y,x ,Nc,Ns,Vsrc,Vsnk,Lt)] = 1/mq*cos(2*M_PI/6);
                              B2_prop_re[prop_index(q,t,jC,jS,iC,iS,y,x ,Nc,Ns,Vsrc,Vsnk,Lt)] = 1/mq*cos(2*M_PI/6);
                              B1_prop_im[prop_index(q,t,jC,jS,iC,iS,y,x ,Nc,Ns,Vsrc,Vsnk,Lt)] = 1/mq*sin(2*M_PI/6);
                              B2_prop_im[prop_index(q,t,jC,jS,iC,iS,y,x ,Nc,Ns,Vsrc,Vsnk,Lt)] = 1/mq*sin(2*M_PI/6);
			                     b_B1_prop_r(y, x, jS, jC, iS, iC, t, q) = 1/mq*cos(2*M_PI/6);
			                     b_B2_prop_r(y, x, jS, jC, iS, iC, t, q) = 1/mq*cos(2*M_PI/6);
			                     b_B1_prop_i(y, x, jS, jC, iS, iC, t, q) = 1/mq*sin(2*M_PI/6);
			                     b_B2_prop_i(y, x, jS, jC, iS, iC, t, q) = 1/mq*sin(2*M_PI/6);
                           }
                           else {
                              B1_prop_re[prop_index(q,t,jC,jS,iC,iS,y,x ,Nc,Ns,Vsrc,Vsnk,Lt)] = 0;
                              B2_prop_re[prop_index(q,t,jC,jS,iC,iS,y,x ,Nc,Ns,Vsrc,Vsnk,Lt)] = 0;
                              B1_prop_im[prop_index(q,t,jC,jS,iC,iS,y,x ,Nc,Ns,Vsrc,Vsnk,Lt)] = 0;
                              B2_prop_im[prop_index(q,t,jC,jS,iC,iS,y,x ,Nc,Ns,Vsrc,Vsnk,Lt)] = 0;
			                     b_B1_prop_r(y, x, jS, jC, iS, iC, t, q) = 0.0;
			                     b_B2_prop_r(y, x, jS, jC, iS, iC, t, q) = 0.0;
			                     b_B1_prop_i(y, x, jS, jC, iS, iC, t, q) = 0.0;
			                     b_B2_prop_i(y, x, jS, jC, iS, iC, t, q) = 0.0;
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
	      double v1 = rand()%10;
	      double v2 = rand()%10;
	      double v3 = rand()%10;
	      double v4 = rand()%10;
         src_psi_B1_re[index_2d(x,m ,Nsrc)] = v1;
         src_psi_B1_im[index_2d(x,m ,Nsrc)] = v2;
         src_psi_B2_re[index_2d(x,m ,Nsrc)] = v3;
         src_psi_B2_im[index_2d(x,m ,Nsrc)] = v4;
         b_B1_src_psi_r(x,m) = v1;
         b_B1_src_psi_i(x,m) = v2;
         b_B2_src_psi_r(x,m) = v3;
         b_B2_src_psi_i(x,m) = v4;
      }
   double* snk_psi_re = (double *) malloc(Vsnk * Vsnk * Nsnk * sizeof (double));
   double* snk_psi_im = (double *) malloc(Vsnk * Vsnk * Nsnk * sizeof (double));
   for (n = 0; n < Nsnk; n++)
      for (x1 = 0; x1 < Vsnk; x1++)
         for (x2 = 0; x2 < Vsnk; x2++) {
	         double v1 = rand()%10;
	         double v2 = rand()%10;
            snk_psi_re[index_3d(x1,x2,n ,Vsnk,Nsnk)] = v1;
            snk_psi_im[index_3d(x1,x2,n ,Vsnk,Nsnk)] = v2;
            b_snk_psi_r(n,x2,x1) = v1;
            b_snk_psi_i(n,x2,x1) = v2;
         }
   double* snk_psi_B1_re = (double *) malloc(Nsnk * Vsnk * sizeof (double));
   double* snk_psi_B1_im = (double *) malloc(Nsnk * Vsnk * sizeof (double));
   double* snk_psi_B2_re = (double *) malloc(Nsnk * Vsnk * sizeof (double));
   double* snk_psi_B2_im = (double *) malloc(Nsnk * Vsnk * sizeof (double));
   for (n = 0; n < Nsnk; n++) {
      for (x = 0; x < Vsnk; x++) {
	      double v1 = rand()%10;
	      double v2 = rand()%10;
	      double v3 = rand()%10;
	      double v4 = rand()%10;
         snk_psi_B1_re[index_2d(x,n ,Nsnk)] = v1;
         snk_psi_B1_im[index_2d(x,n ,Nsnk)] = v2;
         snk_psi_B2_re[index_2d(x,n ,Nsnk)] = v3;
         snk_psi_B2_im[index_2d(x,n ,Nsnk)] = v4;
      }
   }
   double* hex_src_psi_re = (double *) malloc(NsrcHex_f * Vsrc * sizeof (double));
   double* hex_src_psi_im = (double *) malloc(NsrcHex_f * Vsrc * sizeof (double));
   double* hex_snk_psi_re = (double *) malloc(NsnkHex_f * Vsnk * sizeof (double));
   double* hex_snk_psi_im = (double *) malloc(NsnkHex_f * Vsnk * sizeof (double));
   for (k = 0; k < NsrcHex_f; k++) {
      for (y = 0; y < Vsrc; y++) {
	      double v1 = rand()%10;
	      double v2 = rand()%10;
         hex_src_psi_re[index_2d(y,k ,NsrcHex_f)] = v1;
         hex_src_psi_im[index_2d(y,k ,NsrcHex_f)] = v2;
      }
   }
   for (k = 0; k < NsnkHex_f; k++) {
      for (x = 0; x < Vsnk; x++) {
	      double v1 = rand()%10;
	      double v2 = rand()%10;
         hex_snk_psi_re[index_2d(x,k ,NsnkHex_f)] = v1;
         hex_snk_psi_im[index_2d(x,k ,NsnkHex_f)] = v2;
      }
   }
   // Weights
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
   double* overall_weight = (double*) malloc(1 * sizeof (double)); 
   overall_weight[0] = 1.0;
   b_overall_weight(0) = 1.0;
   for (int nB1=0; nB1<Nw; nB1++) {
      for (int nB2=0; nB2<Nw; nB2++) {
         /*snk_weights_1[nB1+Nw*nB2] = 1.0/sqrt(2) * src_weights_r1[nB1]*src_weights_r2[nB2];
         snk_weights_2[nB1+Nw*nB2] = -1.0/sqrt(2) * src_weights_r2[nB1]*src_weights_r1[nB2];
         snk_weights_r1[nB1+Nw*nB2] = src_weights_r1[nB1]*src_weights_r1[nB2];
         snk_weights_r2_1[nB1+Nw*nB2] = 1.0/sqrt(2) * src_weights_r1[nB1]*src_weights_r2[nB2];
         snk_weights_r2_2[nB1+Nw*nB2] = 1.0/sqrt(2) * src_weights_r2[nB1]*src_weights_r1[nB2];
         snk_weights_r3[nB1+Nw*nB2] = src_weights_r2[nB1]*src_weights_r2[nB2];*/
         b_snk_weights(nB1+Nw*nB2) = src_weights_r1_P[nB1]*src_weights_r1_P[nB2];
         for (int nq=0; nq<Nq; nq++) {
/*            // A1g
            snk_color_weights_1[index_3d(0,nB1+Nw*nB2,nq ,Nw2,Nq)] = src_color_weights_r1[index_2d(nB1,nq ,Nq)];
            snk_spin_weights_1[index_3d(0,nB1+Nw*nB2,nq ,Nw2,Nq)] = src_spin_weights_r1[index_2d(nB1,nq ,Nq)];
            snk_color_weights_1[index_3d(1,nB1+Nw*nB2,nq ,Nw2,Nq)] = src_color_weights_r2[index_2d(nB2,nq ,Nq)];
            snk_spin_weights_1[index_3d(1,nB1+Nw*nB2,nq ,Nw2,Nq)] = src_spin_weights_r2[index_2d(nB2,nq ,Nq)];
            snk_color_weights_2[index_3d(0,nB1+Nw*nB2,nq ,Nw2,Nq)] = src_color_weights_r2[index_2d(nB1,nq ,Nq)];
            snk_spin_weights_2[index_3d(0,nB1+Nw*nB2,nq ,Nw2,Nq)] = src_spin_weights_r2[index_2d(nB1,nq ,Nq)];
            snk_color_weights_2[index_3d(1,nB1+Nw*nB2,nq ,Nw2,Nq)] = src_color_weights_r1[index_2d(nB2,nq ,Nq)];
            snk_spin_weights_2[index_3d(1,nB1+Nw*nB2,nq ,Nw2,Nq)] = src_spin_weights_r1[index_2d(nB2,nq ,Nq)];
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
            snk_spin_weights_r3[index_3d(1,nB1+Nw*nB2,nq ,Nw2,Nq)] = src_spin_weights_r2[index_2d(nB2,nq ,Nq)]; */
            b_snk_color_weights(nq,nB1+Nw*nB2,0) = src_color_weights_r1_P[nB1][nq];
            b_snk_spin_weights(nq,nB1+Nw*nB2,0) = src_spin_weights_r1_P[nB1][nq];
            b_snk_color_weights(nq,nB1+Nw*nB2,1) = src_color_weights_r1_P[nB2][nq];
            b_snk_spin_weights(nq,nB1+Nw*nB2,1) = src_spin_weights_r1_P[nB2][nq];
         }
      }
   }
/*    for (int i = 0; i<2; i++)
	    for (int j = 0; j<Nw2; j++)
		    for (int k = 0; k<Nq; k++)
		    {
	          double v1 = rand()%10;
             int vc = rand()%3;
             int vi = rand()%2;
			    snk_color_weights[i*Nw2*Nq + j*Nq + k] = vc;
             b_snk_color_weights(k,j,i) = vc;
			    snk_spin_weights[i*Nw2*Nq + j*Nq + k] = vi;
             b_snk_spin_weights(k,j,i) = vi;
			    snk_weights[j] = v1;
             b_snk_weights(j) = 1.0; 
          printf("%4.9f \n", b_snk_weights(j));} */
   // Permutations
   int perms_array[36][6] = { {1,2,3,4,5,6}, {1, 4, 3, 2, 5, 6}, {1, 6, 3, 2, 5, 4}, {1, 2, 3, 6, 5, 4}, {1, 4, 3, 6, 5, 2}, {1, 6, 3, 4, 5, 2}, {3, 2, 1, 4, 5, 6}, {3, 4, 1, 2, 5, 6}, {3, 6, 1, 2, 5, 4}, {3, 2, 1, 6, 5, 4}, {3, 4, 1, 6, 5, 2}, {3, 6, 1, 4, 5, 2}, {5, 2, 1, 4, 3, 6}, {5, 4, 1, 2, 3, 6}, {5, 6, 1, 2, 3, 4}, {5, 2, 1, 6, 3, 4}, {5, 4, 1, 6, 3, 2}, {5, 6, 1, 4, 3, 2}, {1, 2, 5, 4, 3, 6}, {1, 4, 5, 2, 3, 6}, {1, 6, 5, 2, 3, 4}, {1, 2, 5, 6, 3, 4}, {1, 4, 5, 6, 3, 2}, {1, 6, 5, 4, 3, 2}, {3, 2, 5, 4, 1, 6}, {3, 4, 5, 2, 1, 6}, {3, 6, 5, 2, 1, 4}, {3, 2, 5, 6, 1, 4}, {3, 4, 5, 6, 1, 2}, {3, 6, 5, 4, 1, 2}, {5, 2, 3, 4, 1, 6}, {5, 4, 3, 2, 1, 6}, {5, 6, 3, 2, 1, 4}, {5, 2, 3, 6, 1, 4}, {5, 4, 3, 6, 1, 2}, {5, 6, 3, 4, 1, 2} };
   int sigs_array[36] = {1,-1,1,-1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,1,-1,1,-1,1};
   int* perms = (int *) malloc(Nperms * 2*Nq * sizeof (int));
   int sigs[Nperms];
   int permnum = 0;
   for (int i = 0; i < 36; i++) {
      if (perms_array[i][0] > perms_array[i][2]) {
         continue;
      }
      else if (perms_array[i][3] > perms_array[i][5]) {
         continue;
      }
      else {
         for (int q = 0; q < 2*Nq; q++) {
            perms[index_2d(permnum,q ,2*Nq)] = perms_array[i][q];
            b_perms(q,permnum) = perms_array[i][q];
         }
         sigs[permnum] = sigs_array[i];
         b_sigs(permnum) = sigs_array[i];
         permnum += 1;
      }
   }
   // Correlators
   double* C_re = (double *) malloc(Nr * (Nsrc+NsrcHex_f) * (Nsnk+NsnkHex_f) * Lt * sizeof (double));
   double* C_im = (double *) malloc(Nr * (Nsrc+NsrcHex_f) * (Nsnk+NsnkHex_f) * Lt * sizeof (double));
   for (b=0; b<Nr; b++)
      for (m=0; m<Nsrc+NsrcHex_f; m++)
         for (n=0; n<Nsnk+NsnkHex_f; n++)
            for (t=0; t<Lt; t++) {
               C_re[index_4d(b,m,n,t, Nsrc+NsrcHex_f,Nsnk+NsnkHex_f,Lt)] = 0.0;
               C_im[index_4d(b,m,n,t, Nsrc+NsrcHex_f,Nsnk+NsnkHex_f,Lt)] = 0.0;
               //b_C_r(t,n,m,b) = 0.0;
               //b_C_i(t,n,m,b) = 0.0;
               b_C_r(t,n,m) = 0.0;
               b_C_i(t,n,m) = 0.0;
            }
   int space_symmetric = 0;
   int snk_entangled = 0;

#if RUN_REFERENCE
   std::cout << "Start reference C code." <<  std::endl;
   for (int i = 0; i < nb_tests; i++)
   {
	   std::cout << "Run " << i << "/" << nb_tests <<  std::endl;
	   auto start2 = std::chrono::high_resolution_clock::now();

      make_two_nucleon_2pt(C_re, C_im, B1_prop_re, B1_prop_im, B2_prop_re, B2_prop_im, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, perms, sigs, src_psi_B1_re, src_psi_B1_im, src_psi_B2_re, src_psi_B2_im, snk_psi_re, snk_psi_im, snk_psi_B1_re, snk_psi_B1_im, snk_psi_B2_re, snk_psi_B2_im, hex_src_psi_re, hex_src_psi_im, hex_snk_psi_re, hex_snk_psi_im, space_symmetric, snk_entangled, Nc, Ns, Vsrc, Vsnk, Lt, Nw, Nq, Nsrc, Nsnk, NsrcHex_f, NsnkHex_f, Nperms);

	   auto end2 = std::chrono::high_resolution_clock::now();
	   std::chrono::duration<double,std::milli> duration2 = end2 - start2;
	   duration_vector_2.push_back(duration2);
   }
   std::cout << "End reference C code." <<  std::endl;
#endif

   std::cout << "Start Tiramisu code." <<  std::endl;

   for (int i = 0; i < nb_tests; i++)
   {
      std::cout << "Run " << i << "/" << nb_tests <<  std::endl;
      auto start1 = std::chrono::high_resolution_clock::now();

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

				    b_perms.raw_buffer(),
				    b_sigs.raw_buffer(),
				    b_overall_weight.raw_buffer(),
				    b_snk_color_weights.raw_buffer(),
				    b_snk_spin_weights.raw_buffer(),
				    b_snk_weights.raw_buffer(),
				    b_snk_psi_r.raw_buffer(),
				    b_snk_psi_i.raw_buffer());
       
      auto end1 = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double,std::milli> duration1 = end1 - start1;
      duration_vector_1.push_back(duration1);
   }
    std::cout << "End Tiramisu code." <<  std::endl;

    print_time("performance_CPU.csv", "dibaryon", {"Ref", "Tiramisu"}, {median(duration_vector_2), median(duration_vector_1)});
    std::cout << "\nSpeedup = " << median(duration_vector_2)/median(duration_vector_1) << std::endl;

#if RUN_CHECK
    // Compare outputs.

   for (b=1; b<Nr; b++)
      for (m=0; m<Nsrc+NsrcHex_f; m++)
         for (n=0; n<Nsnk+NsnkHex_f; n++)
            for (t=0; t<Lt; t++) {
               //printf("%4.1f + I (%4.1f) vs %4.1f + I (%4.1f) \n", C_re[index_4d(b,m,n,t, Nsrc+NsrcHex_f,Nsnk+NsnkHex_f,Lt)], C_im[index_4d(b,m,n,t, Nsrc+NsrcHex_f,Nsnk+NsnkHex_f,Lt)], b_C_r(t,n,m,b), b_C_i(t,n,m,b));
               //if ((std::abs(C_re[index_4d(b,m,n,t, Nsrc+NsrcHex_f,Nsnk+NsnkHex_f,Lt)] - b_C_r(t,n,m,b)) >= 0.01) ||
	             //  (std::abs(C_im[index_4d(b,m,n,t, Nsrc+NsrcHex_f,Nsnk+NsnkHex_f,Lt)] - b_C_i(t,n,m,b)) >= 0.01))
               printf("m=%d, n=%d, t=%d: %4.1f + I (%4.1f) vs %4.1f + I (%4.1f) \n", m, n, t, C_re[index_4d(b,m,n,t, Nsrc+NsrcHex_f,Nsnk+NsnkHex_f,Lt)], C_im[index_4d(b,m,n,t, Nsrc+NsrcHex_f,Nsnk+NsnkHex_f,Lt)], b_C_r(t,n,m), b_C_i(t,n,m));
               if ((std::abs(C_re[index_4d(b,m,n,t, Nsrc+NsrcHex_f,Nsnk+NsnkHex_f,Lt)] - b_C_r(t,n,m)) >= 0.01) ||
	               (std::abs(C_im[index_4d(b,m,n,t, Nsrc+NsrcHex_f,Nsnk+NsnkHex_f,Lt)] - b_C_i(t,n,m)) >= 0.01))
	            {
		            std::cout << "Error: different computed values for C_r or C_i!" << std::endl;
		            //exit(1);
	            }
            }

#endif

//    std::cout << "\n\n\033[1;32mSuccess: computed values are equal!\033[0m\n\n" << std::endl;

    return 0;
}
