#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <time.h>

// #include "qblocks_2pt.h"                                       /* DEPS */
#include "tiramisu_wrapper.h"
#include "qblocks_2pt_parameters.h"                                       /* DEPS */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


int main() {

  int Vsrc = P_Vsrc;
  int Vsnk = P_Vsnk;
  int Nt = P_Nt;

  int Nsrc = P_Nsrc;
  int NsrcHex = P_NsrcHex;
  int Nsnk = P_Nsnk;
  int NsnkHex = P_NsnkHex;

  int Nc = P_Nc;
  int Ns = P_Ns;
  int Nq = P_Nq;
  int Nw = P_Nw;
  int Nperms = P_Nperms;

   printf("starting block main\n");
   time_t start,end;
   time (&start);
   double dif;
   int iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, iC, iS, jC, jS, kC, kS, x, x1, x2, t, y, nB1, nB2, q, n, m, wnum, i, k, r, b;
   double mq = P_mq;
   double* B1_prop_re = malloc(Nq * Nt * Nc * Ns * Nc * Ns * Vsnk * Vsrc * sizeof (double));
   double* B1_prop_im = malloc(Nq * Nt * Nc * Ns * Nc * Ns * Vsnk * Vsrc * sizeof (double));
   double* B2_prop_re = malloc(Nq * Nt * Nc * Ns * Nc * Ns * Vsnk * Vsrc * sizeof (double));
   double* B2_prop_im = malloc(Nq * Nt * Nc * Ns * Nc * Ns * Vsnk * Vsrc * sizeof (double));
   for (q = 0; q < Nq; q++) {
      for (t = 0; t < Nt; t++) {
         for (iC = 0; iC < Nc; iC++) {
            for (iS = 0; iS < Ns; iS++) {
               for (jC = 0; jC < Nc; jC++) {
                  for (jS = 0; jS < Ns; jS++) {
                     for (y = 0; y < Vsrc; y++) {
                        for (x = 0; x < Vsnk; x++) {
                           if ((jC == iC) && (jS == iS)) {
                              B1_prop_re[prop_index(q,t,jC,jS,iC,iS,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] = cos(2*M_PI/6)/mq;
                              B2_prop_re[prop_index(q,t,jC,jS,iC,iS,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] = cos(2*M_PI/6)/mq;
                              B1_prop_im[prop_index(q,t,jC,jS,iC,iS,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] = sin(2*M_PI/6)/mq;
                              B2_prop_im[prop_index(q,t,jC,jS,iC,iS,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] = sin(2*M_PI/6)/mq;
                           }
                           else {
                              B1_prop_re[prop_index(q,t,jC,jS,iC,iS,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] = 0;
                              B2_prop_re[prop_index(q,t,jC,jS,iC,iS,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] = 0;
                              B1_prop_im[prop_index(q,t,jC,jS,iC,iS,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] = 0;
                              B2_prop_im[prop_index(q,t,jC,jS,iC,iS,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] = 0;
                           }
                        }
                     }
                  }
               }
            }
         }
      }
   }
   printf("built props\n");
   int B1_G1g_r1_color_weights[9][3] = { {0,1,2}, {0,2,1}, {1,0,2} ,{0,1,2}, {0,2,1}, {1,0,2}, {1,2,0}, {2,1,0}, {2,0,1} };
   int B1_G1g_r1_spin_weights[9][3] = { {0,1,0}, {0,1,0}, {0,1,0}, {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0} };
   double B1_G1g_r1_weights[9] = {-2/ sqrt(2), 2/sqrt(2), 2/sqrt(2), 1/sqrt(2), -1/sqrt(2), -1/sqrt(2), 1/sqrt(2), -1/sqrt(2), 1/sqrt(2)};
   int B1_G1g_r2_color_weights[9][3] = { {0,1,2}, {0,2,1}, {1,0,2} ,{1,2,0}, {2,1,0}, {2,0,1}, {0,1,2}, {0,2,1}, {1,0,2} };
   int B1_G1g_r2_spin_weights[9][3] = { {0,1,1}, {0,1,1}, {0,1,1}, {0,1,1}, {0,1,1}, {0,1,1}, {1,0,1}, {1,0,1}, {1,0,1} };
   double B1_G1g_r2_weights[9] = {1/ sqrt(2), -1/sqrt(2), -1/sqrt(2), 1/sqrt(2), -1/sqrt(2), 1/sqrt(2), -2/sqrt(2), 2/sqrt(2), 2/sqrt(2)};
   int* src_color_weights_r1 = malloc(Nw * Nq * sizeof (int));
   int* src_color_weights_r2 = malloc(Nw * Nq * sizeof (int));
   int* src_spin_weights_r1 = malloc(Nw * Nq * sizeof (int));
   int* src_spin_weights_r2 = malloc(Nw * Nq * sizeof (int));
   double src_weights_r1[Nw];
   double src_weights_r2[Nw];
   for (wnum = 0; wnum < Nw; wnum++) {
      for (q = 0; q < Nq; q++) {
         src_color_weights_r1[index_2d(wnum,q ,Nq)] = B1_G1g_r1_color_weights[wnum][q];
         src_color_weights_r2[index_2d(wnum,q ,Nq)] = B1_G1g_r2_color_weights[wnum][q];
         src_spin_weights_r1[index_2d(wnum,q ,Nq)] = B1_G1g_r1_spin_weights[wnum][q];
         src_spin_weights_r2[index_2d(wnum,q ,Nq)] = B1_G1g_r2_spin_weights[wnum][q];
      }
      src_weights_r1[wnum] = B1_G1g_r1_weights[wnum];
      src_weights_r2[wnum] = B1_G1g_r2_weights[wnum];
   }
   printf("built weights\n");
   int perms_array[36][6] = { {1,2,3,4,5,6}, {1, 4, 3, 2, 5, 6}, {1, 6, 3, 2, 5, 4}, {1, 2, 3, 6, 5, 4}, {1, 4, 3, 6, 5, 2}, {1, 6, 3, 4, 5, 2}, {3, 2, 1, 4, 5, 6}, {3, 4, 1, 2, 5, 6}, {3, 6, 1, 2, 5, 4}, {3, 2, 1, 6, 5, 4}, {3, 4, 1, 6, 5, 2}, {3, 6, 1, 4, 5, 2}, {5, 2, 1, 4, 3, 6}, {5, 4, 1, 2, 3, 6}, {5, 6, 1, 2, 3, 4}, {5, 2, 1, 6, 3, 4}, {5, 4, 1, 6, 3, 2}, {5, 6, 1, 4, 3, 2}, {1, 2, 5, 4, 3, 6}, {1, 4, 5, 2, 3, 6}, {1, 6, 5, 2, 3, 4}, {1, 2, 5, 6, 3, 4}, {1, 4, 5, 6, 3, 2}, {1, 6, 5, 4, 3, 2}, {3, 2, 5, 4, 1, 6}, {3, 4, 5, 2, 1, 6}, {3, 6, 5, 2, 1, 4}, {3, 2, 5, 6, 1, 4}, {3, 4, 5, 6, 1, 2}, {3, 6, 5, 4, 1, 2}, {5, 2, 3, 4, 1, 6}, {5, 4, 3, 2, 1, 6}, {5, 6, 3, 2, 1, 4}, {5, 2, 3, 6, 1, 4}, {5, 4, 3, 6, 1, 2}, {5, 6, 3, 4, 1, 2} };
   int sigs_array[36] = {1,-1,1,-1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,1,-1,1,-1,1};
   int* perms = malloc(Nperms * 2*Nq * sizeof (int));
   int sigs[Nperms];
   int permnum = 0;
   for (i = 0; i < 36; i++) {
      if (perms_array[i][0] > perms_array[i][2]) {
         continue;
      }
      else if (perms_array[i][3] > perms_array[i][5]) {
         continue;
      }
      else {
         for (q = 0; q < 2*Nq; q++) {
            perms[index_2d(permnum,q ,2*Nq)] = perms_array[i][q];
         }
         sigs[permnum] = sigs_array[i];
         permnum += 1;
      }
   }
   printf("read %d perms \n", permnum);
   double* src_psi_B1_re = malloc(Nsrc * Vsrc * sizeof (double));
   double* src_psi_B1_im = malloc(Nsrc * Vsrc * sizeof (double));
   double* src_psi_B2_re = malloc(Nsrc * Vsrc * sizeof (double));
   double* src_psi_B2_im = malloc(Nsrc * Vsrc * sizeof (double));

   for (k = 0; k < Nsrc; k++) {
      for (x = 0; x < Vsrc; x++) {
         src_psi_B1_re[index_2d(x,k ,Nsrc)] = 1.0;
         src_psi_B1_im[index_2d(x,k ,Nsrc)] = 0.0;
         src_psi_B2_re[index_2d(x,k ,Nsrc)] = 1.0;
         src_psi_B2_im[index_2d(x,k ,Nsrc)] = 0.0;
      }
   }
   double* snk_psi_B1_re = malloc(Nsnk * Vsnk * sizeof (double));
   double* snk_psi_B1_im = malloc(Nsnk * Vsnk * sizeof (double));
   double* snk_psi_B2_re = malloc(Nsnk * Vsnk * sizeof (double));
   double* snk_psi_B2_im = malloc(Nsnk * Vsnk * sizeof (double));
   for (k = 0; k < Nsnk; k++) {
      for (x = 0; x < Vsnk; x++) {
         snk_psi_B1_re[index_2d(x,k ,Nsnk)] = 1.0;
         snk_psi_B1_im[index_2d(x,k ,Nsnk)] = 0.0;
         snk_psi_B2_re[index_2d(x,k ,Nsnk)] = 1.0;
         snk_psi_B2_im[index_2d(x,k ,Nsnk)] = 0.0;
      }
   }
   double* hex_src_psi_re = malloc(NsrcHex * Vsrc * sizeof (double));
   double* hex_src_psi_im = malloc(NsrcHex * Vsrc * sizeof (double));
   double* hex_snk_psi_re = malloc(NsnkHex * Vsnk * sizeof (double));
   double* hex_snk_psi_im = malloc(NsnkHex * Vsnk * sizeof (double));
   for (k = 0; k < NsrcHex; k++) {
      for (y = 0; y < Vsrc; y++) {
         hex_src_psi_re[index_2d(y,k ,NsrcHex)] = 1.0;
         hex_src_psi_im[index_2d(y,k ,NsrcHex)] = 0.0;
      }
   }
   for (k = 0; k < NsnkHex; k++) {
      for (x = 0; x < Vsnk; x++) {
         hex_snk_psi_re[index_2d(x,k ,NsnkHex)] = 1.0;
         hex_snk_psi_im[index_2d(x,k ,NsnkHex)] = 0.0;
      }
   }
   double* snk_psi_re = malloc(Nsnk * Vsnk * Vsnk * sizeof (double));
   double* snk_psi_im = malloc(Nsnk * Vsnk * Vsnk * sizeof (double));
   for (k = 0; k < Nsnk; k++) {
      for (x = 0; x < Vsnk; x++) {
         for (y = 0; y < Vsnk; y++) {
            snk_psi_re[index_3d(x,y,k ,Vsnk,Nsnk)] = 1.0;
            snk_psi_im[index_3d(x,y,k ,Vsnk,Nsnk)] = 0.0;
         }
      }
   }
   printf("built wavefunctions\n");
   double* C_B1_re = malloc(2 * NsrcHex * NsnkHex * Nt * sizeof (double));
   double* C_B1_im = malloc(2 * NsrcHex * NsnkHex * Nt * sizeof (double));
   for (r=0; r<2; r++) {
     for (m=0; m<NsrcHex; m++) {
        for (n=0; n<NsnkHex; n++) {
           for (t=0; t<Nt; t++) {
            C_B1_re[index_4d(r,m,n,t ,NsrcHex,NsnkHex,Nt)] = 0.0;
            C_B1_im[index_4d(r,m,n,t ,NsrcHex,NsnkHex,Nt)] = 0.0;
            }
         }
      }
   }
   tiramisu_make_nucleon_2pt(C_B1_re, C_B1_im, B1_prop_re, B1_prop_im, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, hex_src_psi_re, hex_src_psi_im, hex_snk_psi_re, hex_snk_psi_im, Nc, Ns, Vsrc, Vsnk, Nt, Nw, Nq, NsrcHex, NsnkHex);
   printf("B=1 results\n");
   for (m=0; m<NsrcHex; m++) {
      for (n=0; n<NsnkHex; n++) {
         for (t=0; t<Nt; t++) {
            printf("C_B1_G1g_r1[%d,%d,%d] = %4.9e + (%4.9e)I\n", m, n, t, C_B1_re[index_4d(0,m,n,t ,NsrcHex,NsnkHex,Nt)], C_B1_im[index_4d(0,m,n,t ,NsrcHex,NsnkHex,Nt)]);
         }
         for (t=0; t<Nt; t++) {
            printf("C_B1_G1g_r2[%d,%d,%d] = %4.9e + (%4.9e)I\n", m, n, t, C_B1_re[index_4d(1,m,n,t ,NsrcHex,NsnkHex,Nt)], C_B1_im[index_4d(1,m,n,t ,NsrcHex,NsnkHex,Nt)]);
         }
      }
   }
   time (&end);
   dif = difftime (end,start);
   printf("total time %5.3f\n",dif); 
   double* C_re = malloc(4 * (Nsrc+NsrcHex) * (Nsnk+NsnkHex) * Nt * sizeof (double));
   double* C_im = malloc(4 * (Nsrc+NsrcHex) * (Nsnk+NsnkHex) * Nt * sizeof (double));
   for (b=0; b<4; b++) {
      for (m=0; m<Nsrc+NsrcHex; m++) {
         for (n=0; n<Nsnk+NsnkHex; n++) {
            for (t=0; t<Nt; t++) {
               C_re[index_4d(b,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] = 0.0;
               C_im[index_4d(b,m,n,t, Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] = 0.0;
            }
         }
      }
   }
   int space_symmetric = 0;
   int snk_entangled = 0;
   int Nw2Hex = 32;
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
         snk_color_weights_A1[index_2d(wnum,q ,2*Nq)] = 0;
         snk_color_weights_T1_r1[index_2d(wnum,q ,2*Nq)] = 0;
         snk_color_weights_T1_r2[index_2d(wnum,q ,2*Nq)] = 0;
         snk_color_weights_T1_r3[index_2d(wnum,q ,2*Nq)] = 0;
         snk_spin_weights_A1[index_2d(wnum,q ,2*Nq)] = 0;
         snk_spin_weights_T1_r1[index_2d(wnum,q ,2*Nq)] = 0;
         snk_spin_weights_T1_r2[index_2d(wnum,q ,2*Nq)] = 0;
         snk_spin_weights_T1_r3[index_2d(wnum,q ,2*Nq)] = 0;
      }
      snk_weights_A1[wnum] = 0.0;
      snk_weights_T1_r1[wnum] = 0.0;
      snk_weights_T1_r2[wnum] = 0.0;
      snk_weights_T1_r3[wnum] = 0.0;
   }
   printf("starting \n");
   tiramisu_make_two_nucleon_2pt(C_re, C_im, B1_prop_re, B1_prop_im, B2_prop_re, B2_prop_im, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, snk_color_weights_A1, snk_spin_weights_A1, snk_weights_A1, snk_color_weights_T1_r1, snk_spin_weights_T1_r1, snk_weights_T1_r1, snk_color_weights_T1_r2, snk_spin_weights_T1_r2, snk_weights_T1_r2, snk_color_weights_T1_r3, snk_spin_weights_T1_r3, snk_weights_T1_r3, perms, sigs, src_psi_B1_re, src_psi_B1_im, src_psi_B2_re, src_psi_B2_im, snk_psi_re, snk_psi_im, snk_psi_B1_re, snk_psi_B1_im, snk_psi_B2_re, snk_psi_B2_im, hex_src_psi_re, hex_src_psi_im, hex_snk_psi_re, hex_snk_psi_im, Nc, Ns, Vsrc, Vsnk, Nt, Nw, Nq, Nsrc, Nsnk, NsrcHex, NsnkHex, Nperms);
   printf("B=2 results\n");
   int* Vtab = malloc((Nsrc+NsrcHex) * (Nsnk+NsnkHex) * sizeof (int));
   for (m=0; m<Nsrc; m++) {
      for (n=0; n<Nsnk; n++) {
         Vtab[index_2d(m,n,Nsnk+NsnkHex)] = 1;
      }
   }
   for (m=0; m<NsrcHex; m++) {
      for (n=0; n<Nsnk; n++) {
         Vtab[index_2d(Nsrc+m,n,Nsnk+NsnkHex)] = Vsrc;
         //Vtab[index_2d(Nsrc+m,n,Nsnk+NsnkHex)] = 1;
      }
   }
   for (m=0; m<Nsrc; m++) {
      for (n=0; n<NsnkHex; n++) {
         Vtab[index_2d(m,Nsnk+n,Nsnk+NsnkHex)] = Vsnk;
         //Vtab[index_2d(m,Nsnk+n,Nsnk+NsnkHex)] = 1;
      }
   }
   for (m=0; m<NsrcHex; m++) {
      for (n=0; n<NsnkHex; n++) {
         Vtab[index_2d(Nsrc+m,Nsnk+n,Nsnk+NsnkHex)] = Vsrc*Vsnk;
         //Vtab[index_2d(Nsrc+m,Nsnk+n,Nsnk+NsnkHex)] = 1;
      }
   }
   for (m=0; m<Nsrc+NsrcHex; m++) {
      for (n=0; n<Nsnk+NsnkHex; n++) {
         for (t=0; t<Nt; t++) {
            printf("C_B2_A1g[%d,%d,%d] = %4.9e + (%4.9e)I\n", m, n, t, C_re[index_4d(0,m,n,t ,Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)], C_im[index_4d(0,m,n,t ,Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)]);
         }
         for (t=0; t<Nt; t++) {
            printf("C_B2_T1g_r1[%d,%d,%d] = %4.9e + (%4.9e)I\n", m, n, t, C_re[index_4d(1,m,n,t ,Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)], C_im[index_4d(1,m,n,t ,Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)]);
         }
         for (t=0; t<Nt; t++) {
            printf("C_B2_T1g_r2[%d,%d,%d] = %4.9e + (%4.9e)I\n", m, n, t, C_re[index_4d(2,m,n,t ,Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)], C_im[index_4d(2,m,n,t ,Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)]);
         }
         for (t=0; t<Nt; t++) {
            printf("C_B2_T1g_r3[%d,%d,%d] = %4.9e + (%4.9e)I\n", m, n, t, C_re[index_4d(3,m,n,t ,Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)], C_im[index_4d(3,m,n,t ,Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)]);
         }
      }
   }
   printf("B=2 ratios\n");
   for (m=0; m<Nsrc+NsrcHex; m++) {
      for (n=0; n<Nsnk+NsnkHex; n++) {
         for (t=0; t<Nt; t++) {
            printf("R_B2_A1g[%d,%d,%d] = %4.9e \n", m, n, t, Vtab[index_2d(m,n,Nsnk+NsnkHex)]*C_re[index_4d(0,m,n,t ,Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] / (C_B1_re[index_4d(0,0,0,t ,NsrcHex,NsnkHex,Nt)] * C_B1_re[index_4d(1,0,0,t ,NsrcHex,NsnkHex,Nt)]) );
         }
         for (t=0; t<Nt; t++) {
            printf("R_B2_T1g_r1[%d,%d,%d] = %4.9e \n", m, n, t, Vtab[index_2d(m,n,Nsnk+NsnkHex)]*C_re[index_4d(1,m,n,t ,Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] / (C_B1_re[index_4d(0,0,0,t ,NsrcHex,NsnkHex,Nt)] * C_B1_re[index_4d(1,0,0,t ,NsrcHex,NsnkHex,Nt)]) );
         }
         for (t=0; t<Nt; t++) {
            printf("R_B2_T1g_r2[%d,%d,%d] = %4.9e \n", m, n, t, Vtab[index_2d(m,n,Nsnk+NsnkHex)]*C_re[index_4d(2,m,n,t ,Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] / (C_B1_re[index_4d(0,0,0,t ,NsrcHex,NsnkHex,Nt)] * C_B1_re[index_4d(1,0,0,t ,NsrcHex,NsnkHex,Nt)]) );
         }
         for (t=0; t<Nt; t++) {
            printf("R_B2_T1g_r3[%d,%d,%d] = %4.9e \n", m, n, t, Vtab[index_2d(m,n,Nsnk+NsnkHex)]*C_re[index_4d(3,m,n,t ,Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] / (C_B1_re[index_4d(0,0,0,t ,NsrcHex,NsnkHex,Nt)] * C_B1_re[index_4d(1,0,0,t ,NsrcHex,NsnkHex,Nt)]) );
         }
      }
   }
   time (&end);
   dif = difftime (end,start);
   printf("total time %5.3f\n",dif); 
   return 0;
}
