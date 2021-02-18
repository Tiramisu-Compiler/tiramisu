#include <complex>
// started as C code
#include <stdio.h> 
#include <stdlib.h>
#include <math.h>
#include <time.h>

int index_2d(int a, int b, int length2) {
   return b +length2*( a );
}
int index_3d(int a, int b, int c, int length2, int length3) {
   return c +length3*( b +length2*( a ));
}
int index_4d(int a, int b, int c, int d, int length2, int length3, int length4) {
   return d +length4*( c +length3*( b +length2*( a )));
}
int index_5d(int a, int b, int c, int d, int e, int length2, int length3, int length4, int length5) {
   return e + length5*( d +length4*( c +length3*( b +length2*( a ))));
}
int prop_index(int q, int t, int c1, int s1, int c2, int s2, int y, int x, int Nc_f, int Ns_f, int Vsrc_f, int Vsnk_f, int Nt_f) {
   return y +Vsrc_f*( x +Vsnk_f*( s1 +Ns_f*( c1 +Nc_f*( s2 +Ns_f*( c2 +Nc_f*( t +Nt_f* q ))))));
}
int Blocal_index(int c1, int s1, int c2, int s2, int c3, int s3, int m, int Nc_f, int Ns_f, int Nsrc_f) {
   return m +Nsrc_f*( s3 +Ns_f*( c3 +Nc_f*( s2 +Ns_f*( c2 +Nc_f*( s1 +Ns_f*( c1 ))))));
}

void make_local_block(double* Blocal_re, 
    double* Blocal_im, 
    const double* prop_re,
    const double* prop_im, 
    const int* color_weights, 
    const int* spin_weights, 
    const double* weights, 
    const double* psi_re, 
    const double* psi_im,
    const int t,
    const int x,
    const int Nc_f,
    const int Ns_f,
    const int Vsrc_f,
    const int Vsnk_f,
    const int Nt_f,
    const int Nw_f,
    const int Nq_f,
    const int Nsrc_f) {
   /* loop indices */
   int iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, iC, iS, jC, jS, kC, kS, y, wnum, m;
   /* subexpressions */
   std::complex<double> prop_prod_02;
   std::complex<double> prop_prod;
   /* initialize */
   for (iCprime=0; iCprime<Nc_f; iCprime++) {
      for (iSprime=0; iSprime<Ns_f; iSprime++) {
         for (kCprime=0; kCprime<Nc_f; kCprime++) {
            for (kSprime=0; kSprime<Ns_f; kSprime++) {
               for (jCprime=0; jCprime<Nc_f; jCprime++) {
                  for (jSprime=0; jSprime<Ns_f; jSprime++) {
                     for (m=0; m<Nsrc_f; m++) {
                        Blocal_re[Blocal_index(iCprime,iSprime,kCprime,kSprime,jCprime,jSprime,m ,Nc_f,Ns_f,Nsrc_f)] = 0.0;
                        Blocal_im[Blocal_index(iCprime,iSprime,kCprime,kSprime,jCprime,jSprime,m ,Nc_f,Ns_f,Nsrc_f)] = 0.0;
                     }
                  }
               }
            }
         }
      }
   }
   /* build local (no quark exchange) block */
   for (wnum=0; wnum<Nw_f; wnum++) {
      iC = color_weights[index_2d(wnum,0, Nq_f)];
      iS = spin_weights[index_2d(wnum,0, Nq_f)];
      jC = color_weights[index_2d(wnum,1, Nq_f)];
      jS = spin_weights[index_2d(wnum,1, Nq_f)];
      kC = color_weights[index_2d(wnum,2, Nq_f)];
      kS = spin_weights[index_2d(wnum,2, Nq_f)];
      for (iCprime=0; iCprime<Nc_f; iCprime++) {
         for (iSprime=0; iSprime<Ns_f; iSprime++) {
            for (kCprime=0; kCprime<Nc_f; kCprime++) {
               for (kSprime=0; kSprime<Ns_f; kSprime++) {
                  for (y=0; y<Vsrc_f; y++) {
                     std::complex<double> prop_0(prop_re[prop_index(0,t,iC,iS,iCprime,iSprime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)], prop_im[prop_index(0,t,iC,iS,iCprime,iSprime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)]);
                     std::complex<double> prop_2(prop_re[prop_index(2,t,kC,kS,kCprime,kSprime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)], prop_im[prop_index(2,t,kC,kS,kCprime,kSprime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)]);
                     prop_prod_02 = weights[wnum] * ( prop_0 * prop_2 );
                     for (jCprime=0; jCprime<Nc_f; jCprime++) {
                        for (jSprime=0; jSprime<Ns_f; jSprime++) {
                           std::complex<double> prop_1(prop_re[prop_index(1,t,jC,jS,jCprime,jSprime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)], prop_im[prop_index(1,t,jC,jS,jCprime,jSprime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)]);
                           prop_prod = prop_prod_02 * prop_1;
                           for (m=0; m<Nsrc_f; m++) {
                              std::complex<double> psi(psi_re[index_2d(y,m ,Nsrc_f)], psi_im[index_2d(y,m ,Nsrc_f)]);
                              std::complex<double> block = psi * prop_prod;
                              Blocal_re[Blocal_index(iCprime,iSprime,kCprime,kSprime,jCprime,jSprime,m ,Nc_f,Ns_f,Nsrc_f)] += real(block);
                              Blocal_im[Blocal_index(iCprime,iSprime,kCprime,kSprime,jCprime,jSprime,m ,Nc_f,Ns_f,Nsrc_f)] += imag(block);
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

void print_buffer( double* buff, long long size )
{
   for (long long i = 0; i < size; ++i) std::cout << buff[i] << " ";
   std::cout << "\n";
}

double calculate_sum( double* buff, long long size )
{
   double sum = 0.0;
   for (long long i = 0; i < size; ++i) sum += buff[i];
   return sum;
}

void make_nucleon_2pt(double* C_re,
    double* C_im,
    const double* prop_re, 
    const double* prop_im, 
    const int* src_color_weights_r1, 
    const int* src_spin_weights_r1, 
    const double* src_weights_r1, 
    const int* src_color_weights_r2, 
    const int* src_spin_weights_r2, 
    const double* src_weights_r2, 
    const double* src_psi_re,
    const double* src_psi_im, 
    const double* snk_psi_re, 
    const double* snk_psi_im,
    const int Nc_f,
    const int Ns_f,
    const int Vsrc_f,
    const int Vsnk_f,
    const int Nt_f,
    const int Nw_f,
    const int Nq_f,
    const int Nsrc_f,
    const int Nsnk_f) {
   /* indices */
   int n, m, t, x, iC, iS, jC, jS, kC, kS, wnum;
   double* Blocal_r1_re = (double *) malloc(Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsrc_f * sizeof (double));
   double* Blocal_r1_im = (double *) malloc(Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsrc_f * sizeof (double));
   double* Blocal_r2_re = (double *) malloc(Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsrc_f * sizeof (double));
   double* Blocal_r2_im = (double *) malloc(Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsrc_f * sizeof (double));
   
   // printf("--------------------------------------------------\n");
   // printf("Reference(before):\n");
   // std::cout << "Blocal_r1 size: " << (Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsrc_f) << "\n";
   // print_buffer( Blocal_r1_re, 20 );
   // print_buffer( Blocal_r1_im, 20 );
   // printf("Sum Blocal_r1_re: %d \n", calculate_sum(Blocal_r1_re, Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsrc_f));
   // printf("Sum Blocal_r1_im: %d \n", calculate_sum(Blocal_r1_im, Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsrc_f));
   // printf("R--------------------------------------------------\n");

   for (t=0; t<Nt_f; t++) {
      for (x=0; x<Vsnk_f; x++) {
         /* create block */
         make_local_block(Blocal_r1_re, Blocal_r1_im, prop_re, prop_im, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, src_psi_re, src_psi_im, t, x, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f);
         make_local_block(Blocal_r2_re, Blocal_r2_im, prop_re, prop_im, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, src_psi_re, src_psi_im, t, x, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f);
         /* create baryon correlator */
         for (wnum=0; wnum<Nw_f; wnum++) {
            iC = src_color_weights_r1[index_2d(wnum,0, Nq_f)];
            iS = src_spin_weights_r1[index_2d(wnum,0, Nq_f)];
            jC = src_color_weights_r1[index_2d(wnum,1, Nq_f)];
            jS = src_spin_weights_r1[index_2d(wnum,1, Nq_f)];
            kC = src_color_weights_r1[index_2d(wnum,2, Nq_f)];
            kS = src_spin_weights_r1[index_2d(wnum,2, Nq_f)];
            for (m=0; m<Nsrc_f; m++) {
               std::complex<double> block(Blocal_r1_re[Blocal_index(iC,iS,kC,kS,jC,jS,m ,Nc_f,Ns_f,Nsrc_f)], Blocal_r1_im[Blocal_index(iC,iS,kC,kS,jC,jS,m ,Nc_f,Ns_f,Nsrc_f)]);
               std::complex<double> p_block(Blocal_r1_re[Blocal_index(kC,kS,iC,iS,jC,jS,m ,Nc_f,Ns_f,Nsrc_f)], Blocal_r1_im[Blocal_index(kC,kS,iC,iS,jC,jS,m ,Nc_f,Ns_f,Nsrc_f)]);
               for (n=0; n<Nsnk_f; n++) {
                  std::complex<double> psi(snk_psi_re[index_2d( x , n ,Nsnk_f)], snk_psi_im[index_2d( x ,n ,Nsnk_f)]);
                  std::complex<double> corr = (block - p_block) * psi;
                  C_re[index_5d(0,m,0,n,t ,Nsrc_f,2,Nsnk_f,Nt_f)] += src_weights_r1[wnum] * real(corr);
                  C_im[index_5d(0,m,0,n,t ,Nsrc_f,2,Nsnk_f,Nt_f)] += src_weights_r1[wnum] * imag(corr);
               }
               std::complex<double> r_block(Blocal_r2_re[Blocal_index(iC,iS,kC,kS,jC,jS,m ,Nc_f,Ns_f,Nsrc_f)], Blocal_r2_im[Blocal_index(iC,iS,kC,kS,jC,jS,m ,Nc_f,Ns_f,Nsrc_f)]);
               std::complex<double> r_p_block(Blocal_r2_re[Blocal_index(kC,kS,iC,iS,jC,jS,m ,Nc_f,Ns_f,Nsrc_f)], Blocal_r2_im[Blocal_index(kC,kS,iC,iS,jC,jS,m ,Nc_f,Ns_f,Nsrc_f)]);
               for (n=0; n<Nsnk_f; n++) {
                  std::complex<double> psi(snk_psi_re[index_2d( x, n ,Nsnk_f)], snk_psi_im[index_2d( x, n ,Nsnk_f)]);
                  std::complex<double> corr = (r_block - r_p_block) * psi;
                  C_re[index_5d(1,m,0,n,t ,Nsrc_f,2,Nsnk_f,Nt_f)] += src_weights_r1[wnum] * real(corr);
                  C_im[index_5d(1,m,0,n,t ,Nsrc_f,2,Nsnk_f,Nt_f)] += src_weights_r1[wnum] * imag(corr);
               }
            }
         }
         for (wnum=0; wnum<Nw_f; wnum++) {
            iC = src_color_weights_r2[index_2d(wnum,0, Nq_f)];
            iS = src_spin_weights_r2[index_2d(wnum,0, Nq_f)];
            jC = src_color_weights_r2[index_2d(wnum,1, Nq_f)];
            jS = src_spin_weights_r2[index_2d(wnum,1, Nq_f)];
            kC = src_color_weights_r2[index_2d(wnum,2, Nq_f)];
            kS = src_spin_weights_r2[index_2d(wnum,2, Nq_f)];
            for (m=0; m<Nsrc_f; m++) {
               std::complex<double> block(Blocal_r1_re[Blocal_index(iC,iS,kC,kS,jC,jS,m ,Nc_f,Ns_f,Nsrc_f)], Blocal_r1_im[Blocal_index(iC,iS,kC,kS,jC,jS,m ,Nc_f,Ns_f,Nsrc_f)]);
               std::complex<double> p_block(Blocal_r1_re[Blocal_index(kC,kS,iC,iS,jC,jS,m ,Nc_f,Ns_f,Nsrc_f)], Blocal_r1_im[Blocal_index(kC,kS,iC,iS,jC,jS,m ,Nc_f,Ns_f,Nsrc_f)]);
               for (n=0; n<Nsnk_f; n++) {
                  std::complex<double> psi(snk_psi_re[index_2d(x,n ,Nsnk_f)], snk_psi_im[index_2d(x,n ,Nsnk_f)]);
                  std::complex<double> corr = (block - p_block) * psi;
                  C_re[index_5d(0,m,1,n,t ,Nsrc_f,2,Nsnk_f,Nt_f)] += src_weights_r2[wnum] * real(corr);
                  C_im[index_5d(0,m,1,n,t ,Nsrc_f,2,Nsnk_f,Nt_f)] += src_weights_r2[wnum] * imag(corr);
               }
               std::complex<double> r_block(Blocal_r2_re[Blocal_index(iC,iS,kC,kS,jC,jS,m ,Nc_f,Ns_f,Nsrc_f)], Blocal_r2_im[Blocal_index(iC,iS,kC,kS,jC,jS,m ,Nc_f,Ns_f,Nsrc_f)]);
               std::complex<double> r_p_block(Blocal_r2_re[Blocal_index(kC,kS,iC,iS,jC,jS,m ,Nc_f,Ns_f,Nsrc_f)], Blocal_r2_im[Blocal_index(kC,kS,iC,iS,jC,jS,m ,Nc_f,Ns_f,Nsrc_f)]);
               for (n=0; n<Nsnk_f; n++) {
                  std::complex<double> psi(snk_psi_re[index_2d(x,n ,Nsnk_f)], snk_psi_im[index_2d(x,n ,Nsnk_f)]);
                  std::complex<double> corr = (r_block - r_p_block) * psi;
                  C_re[index_5d(1,m,1,n,t ,Nsrc_f,2,Nsnk_f,Nt_f)] += src_weights_r2[wnum] * real(corr);
                  C_im[index_5d(1,m,1,n,t ,Nsrc_f,2,Nsnk_f,Nt_f)] += src_weights_r2[wnum] * imag(corr);
               }
            }
         }
      }
   }

   // printf("--------------------------------------------------\n");
   // printf("Reference(after):\n");
   // std::cout << "Blocal_r1 size: " << (Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsrc_f) << "\n";
   // print_buffer( Blocal_r1_re, Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsrc_f );
   // print_buffer( Blocal_r1_im, Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsrc_f );
   // printf("Sum Blocal_r1_re: %d \n", calculate_sum(Blocal_r1_re, Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsrc_f));
   // printf("Sum Blocal_r1_im: %d \n", calculate_sum(Blocal_r1_im, Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsrc_f));
   // printf("R--------------------------------------------------\n");
   /* clean up */
   free(Blocal_r1_re);
   free(Blocal_r1_im);
   free(Blocal_r2_re);
   free(Blocal_r2_im); 
}
