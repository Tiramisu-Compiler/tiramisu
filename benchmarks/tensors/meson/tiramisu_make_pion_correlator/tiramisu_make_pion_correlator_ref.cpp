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

void make_pion_2pt(double* C_re,
    double* C_im,
    const double* prop_re, 
    const double* prop_im, 
    const int* src_color_weights, 
    const int* src_spin_weights, 
    const double* src_weights, 
    const int* snk_color_weights, 
    const int* snk_spin_weights, 
    const double* snk_weights, 
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
   int n, m, t, x, iC, iS, jC, jS, wnum, wnumSnk, iCprime, iSprime, jCprime, jSprime, y;
   std::complex<double> prop_prod;
   printf("starting pion reference \n");
   for (t=0; t<Nt_f; t++) {
      for (wnum=0; wnum<Nw_f; wnum++) {
         iC = src_color_weights[index_2d(wnum,0, Nq_f)];
         iS = src_spin_weights[index_2d(wnum,0, Nq_f)];
         jC = src_color_weights[index_2d(wnum,1, Nq_f)];
         jS = src_spin_weights[index_2d(wnum,1, Nq_f)];
         for (wnumSnk=0; wnumSnk<Nw_f; wnumSnk++) {
            iCprime = src_color_weights[index_2d(wnumSnk,0, Nq_f)];
            iSprime = src_spin_weights[index_2d(wnumSnk,0, Nq_f)];
            jCprime = src_color_weights[index_2d(wnumSnk,1, Nq_f)];
            jSprime = src_spin_weights[index_2d(wnumSnk,1, Nq_f)];
            for (x=0; x<Vsnk_f; x++) {
               for (y=0; y<Vsrc_f; y++) {
                  std::complex<double> prop_0(prop_re[prop_index(0,t,iC,iS,iCprime,iSprime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)], prop_im[prop_index(0,t,iC,iS,iCprime,iSprime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)]);
                  std::complex<double> prop_1(prop_re[prop_index(1,t,jC,jS,jCprime,jSprime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)], prop_im[prop_index(1,t,jC,jS,jCprime,jSprime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)]);
                  prop_prod = prop_0 * prop_1;
                  for (m=0; m<Nsrc_f; m++) {
                     std::complex<double> src_psi(src_psi_re[index_2d(y,m ,Nsrc_f)], src_psi_im[index_2d(y,m ,Nsrc_f)]);
                     for (n=0; n<Nsnk_f; n++) {
                        std::complex<double> snk_psi(snk_psi_re[index_2d(x,n ,Nsnk_f)], snk_psi_im[index_2d(x,n ,Nsnk_f)]);
                        std::complex<double> corr = prop_prod * src_psi * snk_psi * src_weights[wnum] * src_weights[wnumSnk];
                        C_re[index_5d(0,m,0,n,t ,Nsrc_f,B0Nrows,Nsnk_f,Nt_f)] += real(corr);
                        C_im[index_5d(0,m,0,n,t ,Nsrc_f,B0Nrows,Nsnk_f,Nt_f)] += imag(corr);
                     }
                  }
               }
            }
         }
      }
   } 
}
