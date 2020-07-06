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
                     std::complex<double> prop_0_p(prop_re[prop_index(0,t,iC,iS,kCprime,kSprime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)], prop_im[prop_index(0,t,iC,iS,kCprime,kSprime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)]);
                     std::complex<double> prop_2(prop_re[prop_index(2,t,kC,kS,kCprime,kSprime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)], prop_im[prop_index(2,t,kC,kS,kCprime,kSprime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)]);
                     std::complex<double> prop_2_p(prop_re[prop_index(2,t,kC,kS,iCprime,iSprime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)], prop_im[prop_index(2,t,kC,kS,iCprime,iSprime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)]);
                     prop_prod_02 = weights[wnum] * ( prop_0 * prop_2 - prop_0_p * prop_2_p );
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

void make_local_snk_block(double* Blocal_re, 
    double* Blocal_im, 
    const double* prop_re,
    const double* prop_im, 
    const int* color_weights, 
    const int* spin_weights, 
    const double* weights, 
    const double* psi_re, 
    const double* psi_im,
    const int t,
    const int y,
    const int Nc_f,
    const int Ns_f,
    const int Vsrc_f,
    const int Vsnk_f,
    const int Nt_f,
    const int Nw_f,
    const int Nq_f,
    const int Nsnk_f) {
   /* loop indices */
   int iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, iC, iS, jC, jS, kC, kS, x, wnum, n;
   /* subexpressions */
   double prop_prod_02_re;
   double prop_prod_02_im;
   double prop_prod_re;
   double prop_prod_im;
   /* initialize */
   for (iCprime=0; iCprime<Nc_f; iCprime++) {
      for (iSprime=0; iSprime<Ns_f; iSprime++) {
         for (kCprime=0; kCprime<Nc_f; kCprime++) {
            for (kSprime=0; kSprime<Ns_f; kSprime++) {
               for (jCprime=0; jCprime<Nc_f; jCprime++) {
                  for (jSprime=0; jSprime<Ns_f; jSprime++) {
                     for (n=0; n<Nsnk_f; n++) {
                        Blocal_re[Blocal_index(iCprime,iSprime,kCprime,kSprime,jCprime,jSprime,n ,Nc_f,Ns_f,Nsnk_f)] = 0.0;
                        Blocal_im[Blocal_index(iCprime,iSprime,kCprime,kSprime,jCprime,jSprime,n ,Nc_f,Ns_f,Nsnk_f)] = 0.0;
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
                  for (x=0; x<Vsnk_f; x++) {
                     prop_prod_02_re = weights[wnum] * ( (prop_re[prop_index(0,t,iC,iS,iCprime,iSprime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)] * prop_re[prop_index(2,t,kC,kS,kCprime,kSprime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)] - prop_im[prop_index(0,t,iC,iS,iCprime,iSprime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)] * prop_im[prop_index(2,t,kC,kS,kCprime,kSprime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)]) - (prop_re[prop_index(0,t,iC,iS,kCprime,kSprime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)] * prop_re[prop_index(2,t,kC,kS,iCprime,iSprime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)] - prop_im[prop_index(0,t,iC,iS,kCprime,kSprime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)] * prop_im[prop_index(2,t,kC,kS,iCprime,iSprime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)]) );
                     prop_prod_02_im = weights[wnum] * ( (prop_re[prop_index(0,t,iC,iS,iCprime,iSprime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)] * prop_im[prop_index(2,t,kC,kS,kCprime,kSprime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)] + prop_im[prop_index(0,t,iC,iS,iCprime,iSprime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)] * prop_re[prop_index(2,t,kC,kS,kCprime,kSprime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)]) - (prop_re[prop_index(0,t,iC,iS,kCprime,kSprime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)] * prop_im[prop_index(2,t,kC,kS,iCprime,iSprime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)] + prop_im[prop_index(0,t,iC,iS,kCprime,kSprime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)] * prop_re[prop_index(2,t,kC,kS,iCprime,iSprime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)]) );
                     for (jCprime=0; jCprime<Nc_f; jCprime++) {
                        for (jSprime=0; jSprime<Ns_f; jSprime++) {
                           prop_prod_re = prop_prod_02_re * prop_re[prop_index(1,t,jC,jS,jCprime,jSprime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)] - prop_prod_02_im * prop_im[prop_index(1,t,jC,jS,jCprime,jSprime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)];
                           prop_prod_im = prop_prod_02_re * prop_im[prop_index(1,t,jC,jS,jCprime,jSprime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)] + prop_prod_02_im * prop_re[prop_index(1,t,jC,jS,jCprime,jSprime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)];
                           for (n=0; n<Nsnk_f; n++) {
                              Blocal_re[Blocal_index(iCprime,iSprime,kCprime,kSprime,jCprime,jSprime,n ,Nc_f,Ns_f,Nsnk_f)] += psi_re[index_2d(x,n ,Nsnk_f)] * prop_prod_re - psi_im[index_2d(x,n ,Nsnk_f)] * prop_prod_im;
                              Blocal_im[Blocal_index(iCprime,iSprime,kCprime,kSprime,jCprime,jSprime,n ,Nc_f,Ns_f,Nsnk_f)] += psi_re[index_2d(x,n ,Nsnk_f)] * prop_prod_im + psi_im[index_2d(x,n ,Nsnk_f)] * prop_prod_re;
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

void make_single_block(double* Bsingle_re, 
    double* Bsingle_im, 
    double* Blocal_re, 
    double* Blocal_im, 
    const double* prop_re,
    const double* prop_im, 
    const int* color_weights, 
    const int* spin_weights, 
    const double* weights, 
    const double* psi_re, 
    const double* psi_im,
    const int t,
    const int x1,
    const int x2,
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
   std::complex <double> prop_prod_02;
   std::complex <double> prop_prod;
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
                        Bsingle_re[Blocal_index(iCprime,iSprime,kCprime,kSprime,jCprime,jSprime,m ,Nc_f,Ns_f,Nsrc_f)] = 0.0;
                        Bsingle_im[Blocal_index(iCprime,iSprime,kCprime,kSprime,jCprime,jSprime,m ,Nc_f,Ns_f,Nsrc_f)] = 0.0;
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
                     std::complex<double> prop_0(prop_re[prop_index(0,t,iC,iS,iCprime,iSprime,y,x1 ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)], prop_im[prop_index(0,t,iC,iS,iCprime,iSprime,y,x1 ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)]);
                     std::complex<double> prop_0_p(prop_re[prop_index(0,t,iC,iS,kCprime,kSprime,y,x1 ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)], prop_im[prop_index(0,t,iC,iS,kCprime,kSprime,y,x1 ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)]);
                     std::complex<double> prop_2(prop_re[prop_index(2,t,kC,kS,kCprime,kSprime,y,x1 ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)], prop_im[prop_index(2,t,kC,kS,kCprime,kSprime,y,x1 ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)]);
                     std::complex<double> prop_2_p(prop_re[prop_index(2,t,kC,kS,iCprime,iSprime,y,x1 ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)], prop_im[prop_index(2,t,kC,kS,iCprime,iSprime,y,x1 ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)]);
                     prop_prod_02 = weights[wnum] * ( prop_0 * prop_2 - prop_0_p * prop_2_p );
                     for (jCprime=0; jCprime<Nc_f; jCprime++) {
                        for (jSprime=0; jSprime<Ns_f; jSprime++) {
                           std::complex<double> prop_1(prop_re[prop_index(1,t,jC,jS,jCprime,jSprime,y,x1 ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)], prop_im[prop_index(1,t,jC,jS,jCprime,jSprime,y,x1 ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)]);
                           prop_prod = prop_prod_02 * prop_1;
                           for (m=0; m<Nsrc_f; m++) {
                              std::complex<double> psi(psi_re[index_2d(y,m ,Nsrc_f)], psi_im[index_2d(y,m ,Nsrc_f)]);
                              std::complex<double> block = psi * prop_prod;
                              Blocal_re[Blocal_index(iCprime,iSprime,kCprime,kSprime,jCprime,jSprime,m ,Nc_f,Ns_f,Nsrc_f)] += real(block);
                              Blocal_im[Blocal_index(iCprime,iSprime,kCprime,kSprime,jCprime,jSprime,m ,Nc_f,Ns_f,Nsrc_f)] += imag(block);
                           }
                           std::complex<double> single_prop_1(prop_re[prop_index(1,t,jC,jS,jCprime,jSprime,y,x2 ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)], prop_im[prop_index(1,t,jC,jS,jCprime,jSprime,y,x2 ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)]);
                           prop_prod = prop_prod_02 * single_prop_1;
                           for (m=0; m<Nsrc_f; m++) {
                              std::complex<double> psi(psi_re[index_2d(y,m ,Nsrc_f)], psi_im[index_2d(y,m ,Nsrc_f)]);
                              std::complex<double> block = psi * prop_prod;
                              Bsingle_re[Blocal_index(iCprime,iSprime,kCprime,kSprime,jCprime,jSprime,m ,Nc_f,Ns_f,Nsrc_f)] += real(block);
                              Bsingle_im[Blocal_index(iCprime,iSprime,kCprime,kSprime,jCprime,jSprime,m ,Nc_f,Ns_f,Nsrc_f)] += imag(block);
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

void make_double_block(double* Bdouble_re, 
    double* Bdouble_im, 
    const double* prop_re,
    const double* prop_im, 
    const int* color_weights, 
    const int* spin_weights, 
    const double* weights, 
    const double* psi_re, 
    const double* psi_im,
    const int t,
    const int x1,
    const int x2,
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
   std::complex <double> prop_prod_02;
   std::complex <double> prop_prod;
   /* initialize */
   for (iCprime=0; iCprime<Nc_f; iCprime++) {
      for (iSprime=0; iSprime<Ns_f; iSprime++) {
         for (kCprime=0; kCprime<Nc_f; kCprime++) {
            for (kSprime=0; kSprime<Ns_f; kSprime++) {
               for (jCprime=0; jCprime<Nc_f; jCprime++) {
                  for (jSprime=0; jSprime<Ns_f; jSprime++) {
                     for (m=0; m<Nsrc_f; m++) {
                        Bdouble_re[Blocal_index(iCprime,iSprime,kCprime,kSprime,jCprime,jSprime,m ,Nc_f,Ns_f,Nsrc_f)] = 0.0;
                        Bdouble_im[Blocal_index(iCprime,iSprime,kCprime,kSprime,jCprime,jSprime,m ,Nc_f,Ns_f,Nsrc_f)] = 0.0;
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
                     std::complex<double> prop_0(prop_re[prop_index(0,t,iC,iS,iCprime,iSprime,y,x2 ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)], prop_im[prop_index(0,t,iC,iS,iCprime,iSprime,y,x2 ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)]);
                     std::complex<double> prop_0_p(prop_re[prop_index(0,t,iC,iS,kCprime,kSprime,y,x1 ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)], prop_im[prop_index(0,t,iC,iS,kCprime,kSprime,y,x1 ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)]);
                     std::complex<double> prop_2(prop_re[prop_index(2,t,kC,kS,kCprime,kSprime,y,x1 ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)], prop_im[prop_index(2,t,kC,kS,kCprime,kSprime,y,x1 ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)]);
                     std::complex<double> prop_2_p(prop_re[prop_index(2,t,kC,kS,iCprime,iSprime,y,x2 ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)], prop_im[prop_index(2,t,kC,kS,iCprime,iSprime,y,x2 ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)]);
                     prop_prod_02 = weights[wnum] * ( prop_0 * prop_2 - prop_0_p * prop_2_p );
                     for (jCprime=0; jCprime<Nc_f; jCprime++) {
                        for (jSprime=0; jSprime<Ns_f; jSprime++) {
                           std::complex<double> prop_1(prop_re[prop_index(1,t,jC,jS,jCprime,jSprime,y,x1 ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)], prop_im[prop_index(1,t,jC,jS,jCprime,jSprime,y,x1 ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)]);
                           prop_prod = prop_prod_02 * prop_1;
                           for (m=0; m<Nsrc_f; m++) {
                              std::complex<double> psi(psi_re[index_2d(y,m ,Nsrc_f)], psi_im[index_2d(y,m ,Nsrc_f)]);
                              std::complex<double> block = psi * prop_prod;
                              Bdouble_re[Blocal_index(iCprime,iSprime,kCprime,kSprime,jCprime,jSprime,m ,Nc_f,Ns_f,Nsrc_f)] += real(block);
                              Bdouble_im[Blocal_index(iCprime,iSprime,kCprime,kSprime,jCprime,jSprime,m ,Nc_f,Ns_f,Nsrc_f)] += imag(block);
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

void make_dibaryon_correlator(double* C_re,
    double* C_im,
    const double* B1_Blocal_re, 
    const double* B1_Blocal_im, 
    const double* B1_Bsingle_re, 
    const double* B1_Bsingle_im, 
    const double* B1_Bdouble_re, 
    const double* B1_Bdouble_im, 
    const double* B2_Blocal_re, 
    const double* B2_Blocal_im, 
    const double* B2_Bsingle_re, 
    const double* B2_Bsingle_im, 
    const double* B2_Bdouble_re, 
    const double* B2_Bdouble_im, 
    const int* perms, 
    const int* sigs, 
    const double overall_weight,
    const int* snk_color_weights, 
    const int* snk_spin_weights, 
    const double* snk_weights, 
    const double* snk_psi_re,
    const double* snk_psi_im,
    const int t,
    const int x1,
    const int x2,
    const int Nc_f,
    const int Ns_f,
    const int Vsrc_f,
    const int Vsnk_f,
    const int Nt_f,
    const int Nw_f,
    const int Nq_f,
    const int Nsrc_f,
    const int Nsnk_f,
    const int Nperms_f) {
   /* indices */
   int iC1,iS1,jC1,jS1,kC1,kS1,iC2,iS2,jC2,jS2,kC2,kS2,wnum,nperm,b,n,m;
   int Nb_f = 2;
   int Nw2_f = Nw_f*Nw_f;
   std::complex<double> term, new_term;
   /* build dibaryon */
   int snk_1_nq[Nb_f];
   int snk_2_nq[Nb_f];
   int snk_3_nq[Nb_f];
   int snk_1_b[Nb_f];
   int snk_2_b[Nb_f];
   int snk_3_b[Nb_f];
   int snk_1[Nb_f];
   int snk_2[Nb_f];
   int snk_3[Nb_f];
   for (nperm=0; nperm<Nperms_f; nperm++) {
      for (b=0; b<Nb_f; b++) {
         snk_1[b] = perms[index_2d(nperm,Nq_f*b+0 ,2*Nq_f)] - 1;
         snk_2[b] = perms[index_2d(nperm,Nq_f*b+1 ,2*Nq_f)] - 1;
         snk_3[b] = perms[index_2d(nperm,Nq_f*b+2 ,2*Nq_f)] - 1;
         snk_1_b[b] = (snk_1[b] - snk_1[b] % Nq_f) / Nq_f;
         snk_2_b[b] = (snk_2[b] - snk_2[b] % Nq_f) / Nq_f;
         snk_3_b[b] = (snk_3[b] - snk_3[b] % Nq_f) / Nq_f;
         snk_1_nq[b] = snk_1[b] % Nq_f;
         snk_2_nq[b] = snk_2[b] % Nq_f;
         snk_3_nq[b] = snk_3[b] % Nq_f;
      }
      for (wnum=0; wnum< Nw2_f; wnum++) {
         iC1 = snk_color_weights[index_3d(snk_1_b[0],wnum,snk_1_nq[0] ,Nw2_f,Nq_f)];
         iS1 = snk_spin_weights[index_3d(snk_1_b[0],wnum,snk_1_nq[0] ,Nw2_f,Nq_f)];
         jC1 = snk_color_weights[index_3d(snk_2_b[0],wnum,snk_2_nq[0] ,Nw2_f,Nq_f)];
         jS1 = snk_spin_weights[index_3d(snk_2_b[0],wnum,snk_2_nq[0] ,Nw2_f,Nq_f)];
         kC1 = snk_color_weights[index_3d(snk_3_b[0],wnum,snk_3_nq[0] ,Nw2_f,Nq_f)];
         kS1 = snk_spin_weights[index_3d(snk_3_b[0],wnum,snk_3_nq[0] ,Nw2_f,Nq_f)];
         iC2 = snk_color_weights[index_3d(snk_1_b[1],wnum,snk_1_nq[1] ,Nw2_f,Nq_f)];
         iS2 = snk_spin_weights[index_3d(snk_1_b[1],wnum,snk_1_nq[1] ,Nw2_f,Nq_f)];
         jC2 = snk_color_weights[index_3d(snk_2_b[1],wnum,snk_2_nq[1] ,Nw2_f,Nq_f)];
         jS2 = snk_spin_weights[index_3d(snk_2_b[1],wnum,snk_2_nq[1] ,Nw2_f,Nq_f)];
         kC2 = snk_color_weights[index_3d(snk_3_b[1],wnum,snk_3_nq[1] ,Nw2_f,Nq_f)];
         kS2 = snk_spin_weights[index_3d(snk_3_b[1],wnum,snk_3_nq[1] ,Nw2_f,Nq_f)]; 
         for (m=0; m<Nsrc_f; m++) {
            std::complex<double> term(sigs[nperm] * overall_weight * snk_weights[wnum], 0);
            for (b=0; b<Nb_f; b++) {
               if ((snk_1_b[b] == 0) && (snk_2_b[b] == 0) && (snk_3_b[b] == 0)) {
                  std::complex<double> block(B1_Blocal_re[Blocal_index(iC1,iS1,kC1,kS1,jC1,jS1,m ,Nc_f,Ns_f,Nsrc_f)], B1_Blocal_im[Blocal_index(iC1,iS1,kC1,kS1,jC1,jS1,m ,Nc_f,Ns_f,Nsrc_f)]);
                  new_term = term * block;
               }
               else if ((snk_1_b[b] == 1) && (snk_2_b[b] == 1) && (snk_3_b[b] == 1)) {
                  std::complex<double> block(B2_Blocal_re[Blocal_index(iC2,iS2,kC2,kS2,jC2,jS2,m ,Nc_f,Ns_f,Nsrc_f)], B2_Blocal_im[Blocal_index(iC2,iS2,kC2,kS2,jC2,jS2,m ,Nc_f,Ns_f,Nsrc_f)]);
                  new_term = term * block;
               }
               else if ((snk_1_b[b] == 0) && (snk_3_b[b] == 0)) {
                  std::complex<double> block(B1_Bsingle_re[Blocal_index(iC1,iS1,kC1,kS1,jC1,jS1,m ,Nc_f,Ns_f,Nsrc_f)], B1_Bsingle_im[Blocal_index(iC1,iS1,kC1,kS1,jC1,jS1,m ,Nc_f,Ns_f,Nsrc_f)]);
                  new_term = term * block;
               }
               else if ((snk_1_b[b] == 1) && (snk_3_b[b] == 1)) {
                  std::complex<double> block(B2_Bsingle_re[Blocal_index(iC2,iS2,kC2,kS2,jC2,jS2,m ,Nc_f,Ns_f,Nsrc_f)], B2_Bsingle_im[Blocal_index(iC2,iS2,kC2,kS2,jC2,jS2,m ,Nc_f,Ns_f,Nsrc_f)]);
                  new_term = term * block;
               }
               else if (((snk_1_b[b] == 0) && (snk_2_b[b] == 0)) || ((snk_2_b[b] == 0) && (snk_3_b[b] == 0))) {
                  std::complex<double> block(B1_Bdouble_re[Blocal_index(iC1,iS1,kC1,kS1,jC1,jS1,m ,Nc_f,Ns_f,Nsrc_f)], B1_Bdouble_im[Blocal_index(iC1,iS1,kC1,kS1,jC1,jS1,m ,Nc_f,Ns_f,Nsrc_f)]);
                  new_term = term * block;
               }
               else if (((snk_1_b[b] == 1) && (snk_2_b[b] == 1)) || ((snk_2_b[b] == 1) && (snk_3_b[b] == 1))) {
                  std::complex<double> block(B2_Bdouble_re[Blocal_index(iC2,iS2,kC2,kS2,jC2,jS2,m ,Nc_f,Ns_f,Nsrc_f)], B2_Bdouble_im[Blocal_index(iC2,iS2,kC2,kS2,jC2,jS2,m ,Nc_f,Ns_f,Nsrc_f)]);
                  new_term = term * block;
               }
               term = new_term;
            }
            for (n=0; n<Nsnk_f; n++) {
               std::complex<double> psi(snk_psi_re[index_3d(x1,x2,n ,Vsnk_f,Nsnk_f)], snk_psi_im[index_3d(x1,x2,n ,Vsnk_f,Nsnk_f)]);
               std::complex<double> corr = term * psi;
               C_re[index_3d(m,n,t,Nsnk_f,Nt_f)] += real(corr);
               C_im[index_3d(m,n,t,Nsnk_f,Nt_f)] += imag(corr);
if ((n==0) && (nperm==(Nperms_f-1)) && (wnum==(Nw2_f-1))) {
printf("x=%d, x2=%d, m=%d, n=%d, C=%4.9f +i %4.9f \n", x1, x2, m, n, C_re[index_3d(m,n,t,Nsnk_f,Nt_f)], C_im[index_3d(m,n,t,Nsnk_f,Nt_f)]);
}
            }
         }
      }
   } 
}

void make_dibaryon_hex_correlator(double* C_re,
    double* C_im,
    const double* B1_Blocal_re, 
    const double* B1_Blocal_im, 
    const double* B2_Blocal_re, 
    const double* B2_Blocal_im, 
    const int* perms, 
    const int* sigs, 
    const double overall_weight,
    const int* snk_color_weights, 
    const int* snk_spin_weights, 
    const double* snk_weights, 
    const double* hex_snk_psi_re,
    const double* hex_snk_psi_im,
    const int t,
    const int x,
    const int Nc_f,
    const int Ns_f,
    const int Vsrc_f,
    const int Vsnk_f,
    const int Nt_f,
    const int Nw_f,
    const int Nq_f,
    const int Nsrc_f,
    const int Nsnk_fHex,
    const int Nperms_f) {
   /* indices */
   int iC1,iS1,jC1,jS1,kC1,kS1,iC2,iS2,jC2,jS2,kC2,kS2,wnum,nperm,b,n,m;
   int Nb_f = 2;
   int Nw2_f = Nw_f*Nw_f;
   double term_re, term_im;
   /* build dibaryon */
   int snk_1_nq[Nb_f];
   int snk_2_nq[Nb_f];
   int snk_3_nq[Nb_f];
   int snk_1_b[Nb_f];
   int snk_2_b[Nb_f];
   int snk_3_b[Nb_f];
   int snk_1[Nb_f];
   int snk_2[Nb_f];
   int snk_3[Nb_f];
   for (nperm=0; nperm<Nperms_f; nperm++) {
      for (b=0; b<Nb_f; b++) {
         snk_1[b] = perms[index_2d(nperm,Nq_f*b+0 ,2*Nq_f)] - 1;
         snk_2[b] = perms[index_2d(nperm,Nq_f*b+1 ,2*Nq_f)] - 1;
         snk_3[b] = perms[index_2d(nperm,Nq_f*b+2 ,2*Nq_f)] - 1;
         snk_1_b[b] = (snk_1[b] - snk_1[b] % Nq_f) / Nq_f;
         snk_2_b[b] = (snk_2[b] - snk_2[b] % Nq_f) / Nq_f;
         snk_3_b[b] = (snk_3[b] - snk_3[b] % Nq_f) / Nq_f;
         snk_1_nq[b] = snk_1[b] % Nq_f;
         snk_2_nq[b] = snk_2[b] % Nq_f;
         snk_3_nq[b] = snk_3[b] % Nq_f;
      }
      for (wnum=0; wnum< Nw2_f; wnum++) {
         iC1 = snk_color_weights[index_3d(snk_1_b[0],wnum,snk_1_nq[0] ,Nw2_f,Nq_f)];
         iS1 = snk_spin_weights[index_3d(snk_1_b[0],wnum,snk_1_nq[0] ,Nw2_f,Nq_f)];
         jC1 = snk_color_weights[index_3d(snk_2_b[0],wnum,snk_2_nq[0] ,Nw2_f,Nq_f)];
         jS1 = snk_spin_weights[index_3d(snk_2_b[0],wnum,snk_2_nq[0] ,Nw2_f,Nq_f)];
         kC1 = snk_color_weights[index_3d(snk_3_b[0],wnum,snk_3_nq[0] ,Nw2_f,Nq_f)];
         kS1 = snk_spin_weights[index_3d(snk_3_b[0],wnum,snk_3_nq[0] ,Nw2_f,Nq_f)];
         iC2 = snk_color_weights[index_3d(snk_1_b[1],wnum,snk_1_nq[1] ,Nw2_f,Nq_f)];
         iS2 = snk_spin_weights[index_3d(snk_1_b[1],wnum,snk_1_nq[1] ,Nw2_f,Nq_f)];
         jC2 = snk_color_weights[index_3d(snk_2_b[1],wnum,snk_2_nq[1] ,Nw2_f,Nq_f)];
         jS2 = snk_spin_weights[index_3d(snk_2_b[1],wnum,snk_2_nq[1] ,Nw2_f,Nq_f)];
         kC2 = snk_color_weights[index_3d(snk_3_b[1],wnum,snk_3_nq[1] ,Nw2_f,Nq_f)];
         kS2 = snk_spin_weights[index_3d(snk_3_b[1],wnum,snk_3_nq[1] ,Nw2_f,Nq_f)]; 
         for (m=0; m<Nsrc_f; m++) {
            term_re = sigs[nperm] * overall_weight * snk_weights[wnum] * (B1_Blocal_re[Blocal_index(iC1,iS1,kC1,kS1,jC1,jS1,m ,Nc_f,Ns_f,Nsrc_f)] * B2_Blocal_re[Blocal_index(iC2,iS2,kC2,kS2,jC2,jS2,m ,Nc_f,Ns_f,Nsrc_f)] - B1_Blocal_im[Blocal_index(iC1,iS1,kC1,kS1,jC1,jS1,m ,Nc_f,Ns_f,Nsrc_f)] * B2_Blocal_im[Blocal_index(iC2,iS2,kC2,kS2,jC2,jS2,m ,Nc_f,Ns_f,Nsrc_f)]);
            term_im = sigs[nperm] * overall_weight * snk_weights[wnum] * (B1_Blocal_re[Blocal_index(iC1,iS1,kC1,kS1,jC1,jS1,m ,Nc_f,Ns_f,Nsrc_f)] * B2_Blocal_im[Blocal_index(iC2,iS2,kC2,kS2,jC2,jS2,m ,Nc_f,Ns_f,Nsrc_f)] + B1_Blocal_im[Blocal_index(iC1,iS1,kC1,kS1,jC1,jS1,m ,Nc_f,Ns_f,Nsrc_f)] * B2_Blocal_re[Blocal_index(iC2,iS2,kC2,kS2,jC2,jS2,m ,Nc_f,Ns_f,Nsrc_f)]);
            for (n=0; n<Nsnk_fHex; n++) {
               C_re[index_3d(m,n,t ,Nsnk_fHex,Nt_f)] += hex_snk_psi_re[index_2d(x,n ,Nsnk_fHex)] * term_re - hex_snk_psi_im[index_2d(x,n ,Nsnk_fHex)] * term_im;
               C_im[index_3d(m,n,t ,Nsnk_fHex,Nt_f)] += hex_snk_psi_re[index_2d(x,n ,Nsnk_fHex)] * term_im + hex_snk_psi_im[index_2d(x,n ,Nsnk_fHex)] * term_re;
            }
         }
      }
   } 
}

void make_hex_dibaryon_correlator(double* C_re,
    double* C_im,
    const double* B1_Blocal_re, 
    const double* B1_Blocal_im, 
    const double* B2_Blocal_re, 
    const double* B2_Blocal_im, 
    const int* perms, 
    const int* sigs, 
    const double overall_weight,
    const int* src_color_weights, 
    const int* src_spin_weights, 
    const double* src_weights, 
    const double* hex_src_psi_re,
    const double* hex_src_psi_im,
    const int t,
    const int y,
    const int Nc_f,
    const int Ns_f,
    const int Vsrc_f,
    const int Vsnk_f,
    const int Nt_f,
    const int Nw_f,
    const int Nq_f,
    const int Nsrc_fHex,
    const int Nsnk_f,
    const int Nperms_f) {
   /* indices */
   int iC1,iS1,jC1,jS1,kC1,kS1,iC2,iS2,jC2,jS2,kC2,kS2,wnum,nperm,b,n,m;
   int Nb_f = 2;
   int Nw2_f = Nw_f*Nw_f;
   double term_re, term_im;
   /* build dibaryon */
   int src_1_nq[Nb_f];
   int src_2_nq[Nb_f];
   int src_3_nq[Nb_f];
   int src_1_b[Nb_f];
   int src_2_b[Nb_f];
   int src_3_b[Nb_f];
   int src_1[Nb_f];
   int src_2[Nb_f];
   int src_3[Nb_f];
   for (nperm=0; nperm<Nperms_f; nperm++) {
      for (b=0; b<Nb_f; b++) {
         src_1[b] = perms[index_2d(nperm,Nq_f*b+0 ,2*Nq_f)] - 1;
         src_2[b] = perms[index_2d(nperm,Nq_f*b+1 ,2*Nq_f)] - 1;
         src_3[b] = perms[index_2d(nperm,Nq_f*b+2 ,2*Nq_f)] - 1;
         src_1_b[b] = (src_1[b] - src_1[b] % Nq_f) / Nq_f;
         src_2_b[b] = (src_2[b] - src_2[b] % Nq_f) / Nq_f;
         src_3_b[b] = (src_3[b] - src_3[b] % Nq_f) / Nq_f;
         src_1_nq[b] = src_1[b] % Nq_f;
         src_2_nq[b] = src_2[b] % Nq_f;
         src_3_nq[b] = src_3[b] % Nq_f;
      }
      for (wnum=0; wnum< Nw2_f; wnum++) {
         iC1 = src_color_weights[index_3d(src_1_b[0],wnum,src_1_nq[0] ,Nw2_f,Nq_f)];
         iS1 = src_spin_weights[index_3d(src_1_b[0],wnum,src_1_nq[0] ,Nw2_f,Nq_f)];
         jC1 = src_color_weights[index_3d(src_2_b[0],wnum,src_2_nq[0] ,Nw2_f,Nq_f)];
         jS1 = src_spin_weights[index_3d(src_2_b[0],wnum,src_2_nq[0] ,Nw2_f,Nq_f)];
         kC1 = src_color_weights[index_3d(src_3_b[0],wnum,src_3_nq[0] ,Nw2_f,Nq_f)];
         kS1 = src_spin_weights[index_3d(src_3_b[0],wnum,src_3_nq[0] ,Nw2_f,Nq_f)];
         iC2 = src_color_weights[index_3d(src_1_b[1],wnum,src_1_nq[1] ,Nw2_f,Nq_f)];
         iS2 = src_spin_weights[index_3d(src_1_b[1],wnum,src_1_nq[1] ,Nw2_f,Nq_f)];
         jC2 = src_color_weights[index_3d(src_2_b[1],wnum,src_2_nq[1] ,Nw2_f,Nq_f)];
         jS2 = src_spin_weights[index_3d(src_2_b[1],wnum,src_2_nq[1] ,Nw2_f,Nq_f)];
         kC2 = src_color_weights[index_3d(src_3_b[1],wnum,src_3_nq[1] ,Nw2_f,Nq_f)];
         kS2 = src_spin_weights[index_3d(src_3_b[1],wnum,src_3_nq[1] ,Nw2_f,Nq_f)]; 
         for (n=0; n<Nsnk_f; n++) {
            term_re = sigs[nperm] * overall_weight * src_weights[wnum] * (B1_Blocal_re[Blocal_index(iC1,iS1,kC1,kS1,jC1,jS1,n ,Nc_f,Ns_f,Nsnk_f)] * B2_Blocal_re[Blocal_index(iC2,iS2,kC2,kS2,jC2,jS2,n ,Nc_f,Ns_f,Nsnk_f)] - B1_Blocal_im[Blocal_index(iC1,iS1,kC1,kS1,jC1,jS1,n ,Nc_f,Ns_f,Nsnk_f)] * B2_Blocal_im[Blocal_index(iC2,iS2,kC2,kS2,jC2,jS2,n ,Nc_f,Ns_f,Nsnk_f)]);
            term_im = sigs[nperm] * overall_weight * src_weights[wnum] * (B1_Blocal_re[Blocal_index(iC1,iS1,kC1,kS1,jC1,jS1,n ,Nc_f,Ns_f,Nsnk_f)] * B2_Blocal_im[Blocal_index(iC2,iS2,kC2,kS2,jC2,jS2,n ,Nc_f,Ns_f,Nsnk_f)] + B1_Blocal_im[Blocal_index(iC1,iS1,kC1,kS1,jC1,jS1,n ,Nc_f,Ns_f,Nsnk_f)] * B2_Blocal_re[Blocal_index(iC2,iS2,kC2,kS2,jC2,jS2,n ,Nc_f,Ns_f,Nsnk_f)]);
            for (m=0; m<Nsrc_fHex; m++) {
               C_re[index_3d(m,n,t ,Nsnk_f,Nt_f)] += hex_src_psi_re[index_2d(y,m, Nsrc_fHex)] * term_re - hex_src_psi_im[index_2d(y,m, Nsrc_fHex)] * term_im;
               C_im[index_3d(m,n,t ,Nsnk_f,Nt_f)] += hex_src_psi_re[index_2d(y,m, Nsrc_fHex)] * term_im + hex_src_psi_im[index_2d(y,m, Nsrc_fHex)] * term_re;
            }
         }
      }
   } 
}

void make_hex_correlator(double* C_re,
    double* C_im,
    const double* B1_prop_re, 
    const double* B1_prop_im, 
    const double* B2_prop_re, 
    const double* B2_prop_im, 
    const int* perms, 
    const int* sigs, 
    const int* B1_src_color_weights, 
    const int* B1_src_spin_weights, 
    const double* B1_src_weights, 
    const int* B2_src_color_weights, 
    const int* B2_src_spin_weights, 
    const double* B2_src_weights, 
    const double overall_weight, 
    const int* snk_color_weights, 
    const int* snk_spin_weights, 
    const double* snk_weights, 
    const double* hex_src_psi_re,
    const double* hex_src_psi_im,
    const double* hex_snk_psi_re,
    const double* hex_snk_psi_im,
    const int Nc_f,
    const int Ns_f,
    const int Vsrc_f,
    const int Vsnk_f,
    const int Nt_f,
    const int Nw_f,
    const int Nq_f,
    const int Nsrc_fHex,
    const int Nsnk_fHex,
    const int Nperms_f) {
   /* indices */
   int x,t,wnum,nperm,b,n,m,y,wnumprime;
   int iC1prime,iS1prime,jC1prime,jS1prime,kC1prime,kS1prime,iC2prime,iS2prime,jC2prime,jS2prime,kC2prime,kS2prime;
   int iC1,iS1,jC1,jS1,kC1,kS1,iC2,iS2,jC2,jS2,kC2,kS2;
   int Nb_f = 2;
   int Nw2_f = Nw_f*Nw_f;
   std::complex<double> B1_prop_prod_02, B1_prop_prod, B2_prop_prod_02, B2_prop_prod;
   std::complex<double> prop_prod, new_prop_prod;
   /* build dibaryon */
   int snk_1_nq[Nb_f];
   int snk_2_nq[Nb_f];
   int snk_3_nq[Nb_f];
   int snk_1_b[Nb_f];
   int snk_2_b[Nb_f];
   int snk_3_b[Nb_f];
   int snk_1[Nb_f];
   int snk_2[Nb_f];
   int snk_3[Nb_f];
   for (nperm=0; nperm<Nperms_f; nperm++) {
      for (b=0; b<Nb_f; b++) {
         snk_1[b] = perms[index_2d(nperm,Nq_f*b+0 ,2*Nq_f)] - 1;
         snk_2[b] = perms[index_2d(nperm,Nq_f*b+1 ,2*Nq_f)] - 1;
         snk_3[b] = perms[index_2d(nperm,Nq_f*b+2 ,2*Nq_f)] - 1;
         snk_1_b[b] = (snk_1[b] - snk_1[b] % Nq_f) / Nq_f;
         snk_2_b[b] = (snk_2[b] - snk_2[b] % Nq_f) / Nq_f;
         snk_3_b[b] = (snk_3[b] - snk_3[b] % Nq_f) / Nq_f;
         snk_1_nq[b] = snk_1[b] % Nq_f;
         snk_2_nq[b] = snk_2[b] % Nq_f;
         snk_3_nq[b] = snk_3[b] % Nq_f;
      }
      for (wnumprime=0; wnumprime< Nw2_f; wnumprime++) {
         iC1prime = snk_color_weights[index_3d(snk_1_b[0],wnumprime,snk_1_nq[0] ,Nw2_f,Nq_f)];
         iS1prime = snk_spin_weights[index_3d(snk_1_b[0],wnumprime,snk_1_nq[0] ,Nw2_f,Nq_f)];
         jC1prime = snk_color_weights[index_3d(snk_2_b[0],wnumprime,snk_2_nq[0] ,Nw2_f,Nq_f)];
         jS1prime = snk_spin_weights[index_3d(snk_2_b[0],wnumprime,snk_2_nq[0] ,Nw2_f,Nq_f)];
         kC1prime = snk_color_weights[index_3d(snk_3_b[0],wnumprime,snk_3_nq[0] ,Nw2_f,Nq_f)];
         kS1prime = snk_spin_weights[index_3d(snk_3_b[0],wnumprime,snk_3_nq[0] ,Nw2_f,Nq_f)];
         iC2prime = snk_color_weights[index_3d(snk_1_b[1],wnumprime,snk_1_nq[1] ,Nw2_f,Nq_f)];
         iS2prime = snk_spin_weights[index_3d(snk_1_b[1],wnumprime,snk_1_nq[1] ,Nw2_f,Nq_f)];
         jC2prime = snk_color_weights[index_3d(snk_2_b[1],wnumprime,snk_2_nq[1] ,Nw2_f,Nq_f)];
         jS2prime = snk_spin_weights[index_3d(snk_2_b[1],wnumprime,snk_2_nq[1] ,Nw2_f,Nq_f)];
         kC2prime = snk_color_weights[index_3d(snk_3_b[1],wnumprime,snk_3_nq[1] ,Nw2_f,Nq_f)];
         kS2prime = snk_spin_weights[index_3d(snk_3_b[1],wnumprime,snk_3_nq[1] ,Nw2_f,Nq_f)]; 
         for (t=0; t<Nt_f; t++) {
            for (y=0; y<Vsrc_f; y++) {
               for (x=0; x<Vsnk_f; x++) {
                  std::complex<double> B1_prop_prod_re(0, 0);
                  for (wnum=0; wnum<Nw_f; wnum++) {
                     iC1 = B1_src_color_weights[index_2d(wnum,0, Nq_f)];
                     iS1 = B1_src_spin_weights[index_2d(wnum,0, Nq_f)];
                     jC1 = B1_src_color_weights[index_2d(wnum,1, Nq_f)];
                     jS1 = B1_src_spin_weights[index_2d(wnum,1, Nq_f)];
                     kC1 = B1_src_color_weights[index_2d(wnum,2, Nq_f)];
                     kS1 = B1_src_spin_weights[index_2d(wnum,2, Nq_f)];
                     std::complex<double> B1_prop_0(B1_prop_re[prop_index(0,t,iC1,iS1,iC1prime,iS1prime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)], B1_prop_im[prop_index(0,t,iC1,iS1,iC1prime,iS1prime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)]);
                     std::complex<double> B1_prop_0_p(B1_prop_re[prop_index(0,t,iC1,iS1,kC1prime,kS1prime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)], B1_prop_im[prop_index(0,t,iC1,iS1,kC1prime,kS1prime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)]);
                     std::complex<double> B1_prop_2(B1_prop_re[prop_index(2,t,kC1,kS1,kC1prime,kS1prime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)], B1_prop_im[prop_index(2,t,kC1,kS1,kC1prime,kS1prime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)]);
                     std::complex<double> B1_prop_2_p(B1_prop_re[prop_index(2,t,kC1,kS1,iC1prime,iS1prime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)], B1_prop_im[prop_index(2,t,kC1,kS1,iC1prime,iS1prime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)]);
                     std::complex<double> B1_prop_1(B1_prop_re[prop_index(1,t,jC1,jS1,jC1prime,jS1prime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)], B1_prop_im[prop_index(1,t,jC1,jS1,jC1prime,jS1prime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)]);
                     B1_prop_prod_02 = B1_src_weights[wnum] * ( B1_prop_0 * B1_prop_2 - B1_prop_0_p * B1_prop_2_p );
                     B1_prop_prod += B1_prop_prod_02 * B1_prop_1;
                  }
                  std::complex<double> B2_prop_prod_re(0, 0);
                  for (wnum=0; wnum<Nw_f; wnum++) {
                     iC2 = B2_src_color_weights[index_2d(wnum,0, Nq_f)];
                     iS2 = B2_src_spin_weights[index_2d(wnum,0, Nq_f)];
                     jC2 = B2_src_color_weights[index_2d(wnum,1, Nq_f)];
                     jS2 = B2_src_spin_weights[index_2d(wnum,1, Nq_f)];
                     kC2 = B2_src_color_weights[index_2d(wnum,2, Nq_f)];
                     kS2 = B2_src_spin_weights[index_2d(wnum,2, Nq_f)];
                     std::complex<double> B2_prop_0(B2_prop_re[prop_index(0,t,iC2,iS2,iC2prime,iS2prime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)], B2_prop_im[prop_index(0,t,iC2,iS2,iC2prime,iS2prime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)]);
                     std::complex<double> B2_prop_0_p(B2_prop_re[prop_index(0,t,iC2,iS2,kC2prime,kS2prime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)], B2_prop_im[prop_index(0,t,iC2,iS2,kC2prime,kS2prime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)]);
                     std::complex<double> B2_prop_2(B2_prop_re[prop_index(2,t,kC2,kS2,kC2prime,kS2prime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)], B2_prop_im[prop_index(2,t,kC2,kS2,kC2prime,kS2prime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)]);
                     std::complex<double> B2_prop_2_p(B2_prop_re[prop_index(2,t,kC2,kS2,iC2prime,iS2prime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)], B2_prop_im[prop_index(2,t,kC2,kS2,iC2prime,iS2prime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)]);
                     std::complex<double> B2_prop_1(B2_prop_re[prop_index(1,t,jC2,jS2,jC2prime,jS2prime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)], B2_prop_im[prop_index(1,t,jC2,jS2,jC2prime,jS2prime,y,x ,Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f)]);
                     B2_prop_prod_02 = B2_src_weights[wnum] * ( B2_prop_0 * B2_prop_2 - B2_prop_0_p * B2_prop_2_p );
                     B2_prop_prod += B2_prop_prod_02 * B2_prop_1;
                  }
                  prop_prod = overall_weight * sigs[nperm] * snk_weights[wnumprime] * B1_prop_prod * B2_prop_prod;
                  for (m=0; m<Nsrc_fHex; m++) {
                     std::complex<double> src_psi(hex_src_psi_re[index_2d(y,m ,Nsrc_fHex)],  hex_src_psi_re[index_2d(y,m ,Nsrc_fHex)]);
                     new_prop_prod = prop_prod * src_psi;
                     for (n=0; n<Nsnk_fHex; n++) {
                        std::complex<double> snk_psi(hex_snk_psi_re[index_2d(x,n ,Nsnk_fHex)], hex_snk_psi_im[index_2d(x,n ,Nsnk_fHex)]);
                        std::complex<double> corr = new_prop_prod * snk_psi;
                        C_re[index_3d(m,n,t ,Nsnk_fHex,Nt_f)] += real(corr);
                        C_im[index_3d(m,n,t ,Nsnk_fHex,Nt_f)] += imag(corr);
                     }
                  }
               }
            }
         }
      }
   }
}

void make_two_nucleon_2pt(double* C_re,
    double* C_im,
    const double* B1_prop_re, 
    const double* B1_prop_im, 
    const double* B2_prop_re, 
    const double* B2_prop_im, 
    const int* src_color_weights_r1, 
    const int* src_spin_weights_r1, 
    const double* src_weights_r1, 
    const int* src_color_weights_r2, 
    const int* src_spin_weights_r2, 
    const double* src_weights_r2, 
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
    const int space_symmetric,
    const int snk_entangled,
    const int Nc_f,
    const int Ns_f,
    const int Vsrc_f,
    const int Vsnk_f,
    const int Nt_f,
    const int Nw_f,
    const int Nq_f,
    const int Nsrc_f,
    const int Nsnk_f,
    const int Nsrc_fHex,
    const int Nsnk_fHex,
    const int Nperms_f) {
   /* indices */
   int nB1, nB2, nq, n, m, t, x1, x2, x, y;
   // hold results for two nucleon correlators 
   double* BB_0_re = (double *) malloc(Nsrc_f * Nsnk_f * Nt_f * sizeof (double));
   double* BB_0_im = (double *) malloc(Nsrc_f * Nsnk_f * Nt_f * sizeof (double));
   double* BB_r1_re = (double *) malloc(Nsrc_f * Nsnk_f * Nt_f * sizeof (double));
   double* BB_r1_im = (double *) malloc(Nsrc_f * Nsnk_f * Nt_f * sizeof (double));
   double* BB_r2_re = (double *) malloc(Nsrc_f * Nsnk_f * Nt_f * sizeof (double));
   double* BB_r2_im = (double *) malloc(Nsrc_f * Nsnk_f * Nt_f * sizeof (double));
   double* BB_r3_re = (double *) malloc(Nsrc_f * Nsnk_f * Nt_f * sizeof (double));
   double* BB_r3_im = (double *) malloc(Nsrc_f * Nsnk_f * Nt_f * sizeof (double));
   for (m=0; m<Nsrc_f; m++) {
      for (n=0; n<Nsnk_f; n++) {
         for (t=0; t<Nt_f; t++) {
            BB_0_re[index_3d(m,n,t,Nsnk_f,Nt_f)] = 0.0;
            BB_0_im[index_3d(m,n,t,Nsnk_f,Nt_f)] = 0.0;
            BB_r1_re[index_3d(m,n,t,Nsnk_f,Nt_f)] = 0.0;
            BB_r1_im[index_3d(m,n,t,Nsnk_f,Nt_f)] = 0.0;
            BB_r2_re[index_3d(m,n,t,Nsnk_f,Nt_f)] = 0.0;
            BB_r2_im[index_3d(m,n,t,Nsnk_f,Nt_f)] = 0.0;
            BB_r3_re[index_3d(m,n,t,Nsnk_f,Nt_f)] = 0.0;
            BB_r3_im[index_3d(m,n,t,Nsnk_f,Nt_f)] = 0.0;
         }
      }
   }
   double* BB_H_0_re = (double *) malloc(Nsrc_f * Nsnk_fHex * Nt_f * sizeof (double));
   double* BB_H_0_im = (double *) malloc(Nsrc_f * Nsnk_fHex * Nt_f * sizeof (double));
   double* BB_H_r1_re = (double *) malloc(Nsrc_f * Nsnk_fHex * Nt_f * sizeof (double));
   double* BB_H_r1_im = (double *) malloc(Nsrc_f * Nsnk_fHex * Nt_f * sizeof (double));
   double* BB_H_r2_re = (double *) malloc(Nsrc_f * Nsnk_fHex * Nt_f * sizeof (double));
   double* BB_H_r2_im = (double *) malloc(Nsrc_f * Nsnk_fHex * Nt_f * sizeof (double));
   double* BB_H_r3_re = (double *) malloc(Nsrc_f * Nsnk_fHex * Nt_f * sizeof (double));
   double* BB_H_r3_im = (double *) malloc(Nsrc_f * Nsnk_fHex * Nt_f * sizeof (double));
   for (m=0; m<Nsrc_f; m++) {
      for (n=0; n<Nsnk_fHex; n++) {
         for (t=0; t<Nt_f; t++) {
            BB_H_0_re[index_3d(m,n,t ,Nsnk_fHex,Nt_f)] = 0.0;
            BB_H_0_im[index_3d(m,n,t ,Nsnk_fHex,Nt_f)] = 0.0;
            BB_H_r1_re[index_3d(m,n,t ,Nsnk_fHex,Nt_f)] = 0.0;
            BB_H_r1_im[index_3d(m,n,t ,Nsnk_fHex,Nt_f)] = 0.0;
            BB_H_r2_re[index_3d(m,n,t ,Nsnk_fHex,Nt_f)] = 0.0;
            BB_H_r2_im[index_3d(m,n,t ,Nsnk_fHex,Nt_f)] = 0.0;
            BB_H_r3_re[index_3d(m,n,t ,Nsnk_fHex,Nt_f)] = 0.0;
            BB_H_r3_im[index_3d(m,n,t ,Nsnk_fHex,Nt_f)] = 0.0;
         }
      }
   }
   double* H_BB_0_re = (double *) malloc(Nsrc_fHex * Nsnk_f * Nt_f * sizeof (double));
   double* H_BB_0_im = (double *) malloc(Nsrc_fHex * Nsnk_f * Nt_f * sizeof (double));
   double* H_BB_r1_re = (double *) malloc(Nsrc_fHex * Nsnk_f * Nt_f * sizeof (double));
   double* H_BB_r1_im = (double *) malloc(Nsrc_fHex * Nsnk_f * Nt_f * sizeof (double));
   double* H_BB_r2_re = (double *) malloc(Nsrc_fHex * Nsnk_f * Nt_f * sizeof (double));
   double* H_BB_r2_im = (double *) malloc(Nsrc_fHex * Nsnk_f * Nt_f * sizeof (double));
   double* H_BB_r3_re = (double *) malloc(Nsrc_fHex * Nsnk_f * Nt_f * sizeof (double));
   double* H_BB_r3_im = (double *) malloc(Nsrc_fHex * Nsnk_f * Nt_f * sizeof (double));
   for (m=0; m<Nsrc_fHex; m++) {
      for (n=0; n<Nsnk_f; n++) {
         for (t=0; t<Nt_f; t++) {
            H_BB_0_re[index_3d(m,n,t,Nsnk_f,Nt_f)] = 0.0;
            H_BB_0_im[index_3d(m,n,t,Nsnk_f,Nt_f)] = 0.0;
            H_BB_r1_re[index_3d(m,n,t,Nsnk_f,Nt_f)] = 0.0;
            H_BB_r1_im[index_3d(m,n,t,Nsnk_f,Nt_f)] = 0.0;
            H_BB_r2_re[index_3d(m,n,t,Nsnk_f,Nt_f)] = 0.0;
            H_BB_r2_im[index_3d(m,n,t,Nsnk_f,Nt_f)] = 0.0;
            H_BB_r3_re[index_3d(m,n,t,Nsnk_f,Nt_f)] = 0.0;
            H_BB_r3_im[index_3d(m,n,t,Nsnk_f,Nt_f)] = 0.0; 
         }
      }
   }
   double* H_0_re = (double *) malloc(Nsrc_fHex * Nsnk_fHex * Nt_f * sizeof (double));
   double* H_0_im = (double *) malloc(Nsrc_fHex * Nsnk_fHex * Nt_f * sizeof (double));
   double* H_r1_re = (double *) malloc(Nsrc_fHex * Nsnk_fHex * Nt_f * sizeof (double));
   double* H_r1_im = (double *) malloc(Nsrc_fHex * Nsnk_fHex * Nt_f * sizeof (double));
   double* H_r2_re = (double *) malloc(Nsrc_fHex * Nsnk_fHex * Nt_f * sizeof (double));
   double* H_r2_im = (double *) malloc(Nsrc_fHex * Nsnk_fHex * Nt_f * sizeof (double));
   double* H_r3_re = (double *) malloc(Nsrc_fHex * Nsnk_fHex * Nt_f * sizeof (double));
   double* H_r3_im = (double *) malloc(Nsrc_fHex * Nsnk_fHex * Nt_f * sizeof (double));
   for (m=0; m<Nsrc_fHex; m++) {
      for (n=0; n<Nsnk_fHex; n++) {
         for (t=0; t<Nt_f; t++) {
            H_0_re[index_3d(m,n,t ,Nsnk_fHex,Nt_f)] = 0.0;
            H_0_im[index_3d(m,n,t ,Nsnk_fHex,Nt_f)] = 0.0;
            H_r1_re[index_3d(m,n,t ,Nsnk_fHex,Nt_f)] = 0.0;
            H_r1_im[index_3d(m,n,t ,Nsnk_fHex,Nt_f)] = 0.0;
            H_r2_re[index_3d(m,n,t ,Nsnk_fHex,Nt_f)] = 0.0;
            H_r2_im[index_3d(m,n,t ,Nsnk_fHex,Nt_f)] = 0.0;
            H_r3_re[index_3d(m,n,t ,Nsnk_fHex,Nt_f)] = 0.0;
            H_r3_im[index_3d(m,n,t ,Nsnk_fHex,Nt_f)] = 0.0;
         }
      }
   }
   /* compute two nucleon snk weights */
   int Nw2_f = Nw_f*Nw_f;
   int* snk_color_weights_1 = (int *) malloc(2 * Nw2_f * Nq_f * sizeof (int));
   int* snk_color_weights_2 = (int *) malloc(2 * Nw2_f * Nq_f * sizeof (int));
   int* snk_color_weights_r1 = (int *) malloc(2 * Nw2_f * Nq_f * sizeof (int));
   int* snk_color_weights_r2_1 = (int *) malloc(2 * Nw2_f * Nq_f * sizeof (int));
   int* snk_color_weights_r2_2 = (int *) malloc(2 * Nw2_f * Nq_f * sizeof (int));
   int* snk_color_weights_r3 = (int *) malloc(2 * Nw2_f * Nq_f * sizeof (int));
   int* snk_spin_weights_1 = (int *) malloc(2 * Nw2_f * Nq_f * sizeof (int));
   int* snk_spin_weights_2 = (int *) malloc(2 * Nw2_f * Nq_f * sizeof (int));
   int* snk_spin_weights_r1 = (int *) malloc(2 * Nw2_f * Nq_f * sizeof (int));
   int* snk_spin_weights_r2_1 = (int *) malloc(2 * Nw2_f * Nq_f * sizeof (int));
   int* snk_spin_weights_r2_2 = (int *) malloc(2 * Nw2_f * Nq_f * sizeof (int));
   int* snk_spin_weights_r3 = (int *) malloc(2 * Nw2_f * Nq_f * sizeof (int));
   double* snk_weights_1 = (double *) malloc(Nw2_f * sizeof (double));
   double* snk_weights_2 = (double *) malloc(Nw2_f * sizeof (double));
   double* snk_weights_r1 = (double *) malloc(Nw2_f * sizeof (double));
   double* snk_weights_r2_1 = (double *) malloc(Nw2_f * sizeof (double));
   double* snk_weights_r2_2 = (double *) malloc(Nw2_f * sizeof (double));
   double* snk_weights_r3 = (double *) malloc(Nw2_f * sizeof (double));
   for (nB1=0; nB1<Nw_f; nB1++) {
      for (nB2=0; nB2<Nw_f; nB2++) {
         snk_weights_1[nB1+Nw_f*nB2] = 1.0/sqrt(2) * src_weights_r1[nB1]*src_weights_r2[nB2];
         snk_weights_2[nB1+Nw_f*nB2] = -1.0/sqrt(2) * src_weights_r2[nB1]*src_weights_r1[nB2];
         snk_weights_r1[nB1+Nw_f*nB2] = src_weights_r1[nB1]*src_weights_r1[nB2];
         snk_weights_r2_1[nB1+Nw_f*nB2] = 1.0/sqrt(2) * src_weights_r1[nB1]*src_weights_r2[nB2];
         snk_weights_r2_2[nB1+Nw_f*nB2] = 1.0/sqrt(2) * src_weights_r2[nB1]*src_weights_r1[nB2];
         snk_weights_r3[nB1+Nw_f*nB2] = src_weights_r2[nB1]*src_weights_r2[nB2];
         for (nq=0; nq<Nq_f; nq++) {
            // A1g
            snk_color_weights_1[index_3d(0,nB1+Nw_f*nB2,nq ,Nw2_f,Nq_f)] = src_color_weights_r1[index_2d(nB1,nq ,Nq_f)];
            snk_spin_weights_1[index_3d(0,nB1+Nw_f*nB2,nq ,Nw2_f,Nq_f)] = src_spin_weights_r1[index_2d(nB1,nq ,Nq_f)];
            snk_color_weights_1[index_3d(1,nB1+Nw_f*nB2,nq ,Nw2_f,Nq_f)] = src_color_weights_r2[index_2d(nB2,nq ,Nq_f)];
            snk_spin_weights_1[index_3d(1,nB1+Nw_f*nB2,nq ,Nw2_f,Nq_f)] = src_spin_weights_r2[index_2d(nB2,nq ,Nq_f)];
            snk_color_weights_2[index_3d(0,nB1+Nw_f*nB2,nq ,Nw2_f,Nq_f)] = src_color_weights_r2[index_2d(nB1,nq ,Nq_f)];
            snk_spin_weights_2[index_3d(0,nB1+Nw_f*nB2,nq ,Nw2_f,Nq_f)] = src_spin_weights_r2[index_2d(nB1,nq ,Nq_f)];
            snk_color_weights_2[index_3d(1,nB1+Nw_f*nB2,nq ,Nw2_f,Nq_f)] = src_color_weights_r1[index_2d(nB2,nq ,Nq_f)];
            snk_spin_weights_2[index_3d(1,nB1+Nw_f*nB2,nq ,Nw2_f,Nq_f)] = src_spin_weights_r1[index_2d(nB2,nq ,Nq_f)];
            // T1g_r1
            snk_color_weights_r1[index_3d(0,nB1+Nw_f*nB2,nq ,Nw2_f,Nq_f)] = src_color_weights_r1[index_2d(nB1,nq ,Nq_f)];
            snk_spin_weights_r1[index_3d(0,nB1+Nw_f*nB2,nq ,Nw2_f,Nq_f)] = src_spin_weights_r1[index_2d(nB1,nq ,Nq_f)];
            snk_color_weights_r1[index_3d(1,nB1+Nw_f*nB2,nq ,Nw2_f,Nq_f)] = src_color_weights_r1[index_2d(nB2,nq ,Nq_f)];
            snk_spin_weights_r1[index_3d(1,nB1+Nw_f*nB2,nq ,Nw2_f,Nq_f)] = src_spin_weights_r1[index_2d(nB2,nq ,Nq_f)];
            // T1g_r2
            snk_color_weights_r2_1[index_3d(0,nB1+Nw_f*nB2,nq ,Nw2_f,Nq_f)] = src_color_weights_r1[index_2d(nB1,nq ,Nq_f)];
            snk_spin_weights_r2_1[index_3d(0,nB1+Nw_f*nB2,nq ,Nw2_f,Nq_f)] = src_spin_weights_r1[index_2d(nB1,nq ,Nq_f)];
            snk_color_weights_r2_1[index_3d(1,nB1+Nw_f*nB2,nq ,Nw2_f,Nq_f)] = src_color_weights_r2[index_2d(nB2,nq ,Nq_f)];
            snk_spin_weights_r2_1[index_3d(1,nB1+Nw_f*nB2,nq ,Nw2_f,Nq_f)] = src_spin_weights_r2[index_2d(nB2,nq ,Nq_f)];
            snk_color_weights_r2_2[index_3d(0,nB1+Nw_f*nB2,nq ,Nw2_f,Nq_f)] = src_color_weights_r2[index_2d(nB1,nq ,Nq_f)];
            snk_spin_weights_r2_2[index_3d(0,nB1+Nw_f*nB2,nq ,Nw2_f,Nq_f)] = src_spin_weights_r2[index_2d(nB1,nq ,Nq_f)];
            snk_color_weights_r2_2[index_3d(1,nB1+Nw_f*nB2,nq ,Nw2_f,Nq_f)] = src_color_weights_r1[index_2d(nB2,nq ,Nq_f)];
            snk_spin_weights_r2_2[index_3d(1,nB1+Nw_f*nB2,nq ,Nw2_f,Nq_f)] = src_spin_weights_r1[index_2d(nB2,nq ,Nq_f)];
            // T1g_r3 
            snk_color_weights_r3[index_3d(0,nB1+Nw_f*nB2,nq ,Nw2_f,Nq_f)] = src_color_weights_r2[index_2d(nB1,nq ,Nq_f)];
            snk_spin_weights_r3[index_3d(0,nB1+Nw_f*nB2,nq ,Nw2_f,Nq_f)] = src_spin_weights_r2[index_2d(nB1,nq ,Nq_f)];
            snk_color_weights_r3[index_3d(1,nB1+Nw_f*nB2,nq ,Nw2_f,Nq_f)] = src_color_weights_r2[index_2d(nB2,nq ,Nq_f)];
            snk_spin_weights_r3[index_3d(1,nB1+Nw_f*nB2,nq ,Nw2_f,Nq_f)] = src_spin_weights_r2[index_2d(nB2,nq ,Nq_f)];
         }
      }
   }
   double overall_weight;
   if (space_symmetric == 0) {
      overall_weight = 1.0;
   }
   else {
      overall_weight = 1.0/2.0;
   } 
   printf("made snk weights \n");
   if (Nsrc_f > 0 && Nsnk_f > 0) {
         /* BB_BB */
      double* B1_Blocal_r1_re = (double *) malloc(Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsrc_f * sizeof (double));
      double* B1_Blocal_r1_im = (double *) malloc(Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsrc_f * sizeof (double));
      double* B1_Blocal_r2_re = (double *) malloc(Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsrc_f * sizeof (double));
      double* B1_Blocal_r2_im = (double *) malloc(Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsrc_f * sizeof (double));
      double* B2_Blocal_r1_re = (double *) malloc(Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsrc_f * sizeof (double));
      double* B2_Blocal_r1_im = (double *) malloc(Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsrc_f * sizeof (double));
      double* B2_Blocal_r2_re = (double *) malloc(Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsrc_f * sizeof (double));
      double* B2_Blocal_r2_im = (double *) malloc(Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsrc_f * sizeof (double));
      double* B1_Bsingle_r1_re = (double *) malloc(Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsrc_f * sizeof (double));
      double* B1_Bsingle_r1_im = (double *) malloc(Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsrc_f * sizeof (double));
      double* B1_Bsingle_r2_re = (double *) malloc(Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsrc_f * sizeof (double));
      double* B1_Bsingle_r2_im = (double *) malloc(Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsrc_f * sizeof (double));
      double* B2_Bsingle_r1_re = (double *) malloc(Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsrc_f * sizeof (double));
      double* B2_Bsingle_r1_im = (double *) malloc(Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsrc_f * sizeof (double));
      double* B2_Bsingle_r2_re = (double *) malloc(Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsrc_f * sizeof (double));
      double* B2_Bsingle_r2_im = (double *) malloc(Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsrc_f * sizeof (double));
      double* B1_Bdouble_r1_re = (double *) malloc(Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsrc_f * sizeof (double));
      double* B1_Bdouble_r1_im = (double *) malloc(Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsrc_f * sizeof (double));
      double* B1_Bdouble_r2_re = (double *) malloc(Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsrc_f * sizeof (double));
      double* B1_Bdouble_r2_im = (double *) malloc(Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsrc_f * sizeof (double));
      double* B2_Bdouble_r1_re = (double *) malloc(Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsrc_f * sizeof (double));
      double* B2_Bdouble_r1_im = (double *) malloc(Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsrc_f * sizeof (double));
      double* B2_Bdouble_r2_re = (double *) malloc(Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsrc_f * sizeof (double));
      double* B2_Bdouble_r2_im = (double *) malloc(Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsrc_f * sizeof (double));
      for (t=0; t<Nt_f; t++) {
         for (x1 =0; x1<Vsnk_f; x1++) {
            for (x2 =0; x2<Vsnk_f; x2++) {
               // create blocks
               make_single_block(B1_Bsingle_r1_re, B1_Bsingle_r1_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B1_prop_re, B1_prop_im, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, src_psi_B1_re, src_psi_B1_im, t, x1, x2, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f);
               make_single_block(B1_Bsingle_r2_re, B1_Bsingle_r2_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B1_prop_re, B1_prop_im, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, src_psi_B1_re, src_psi_B1_im, t, x1, x2, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f);
               make_double_block(B1_Bdouble_r1_re, B1_Bdouble_r1_im, B1_prop_re, B1_prop_im, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, src_psi_B1_re, src_psi_B1_im, t, x1, x2, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f);
               make_double_block(B1_Bdouble_r2_re, B1_Bdouble_r2_im, B1_prop_re, B1_prop_im, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, src_psi_B1_re, src_psi_B1_im, t, x1, x2, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f);
               make_single_block(B2_Bsingle_r1_re, B2_Bsingle_r1_im, B2_Blocal_r1_re, B2_Blocal_r1_im, B2_prop_re, B2_prop_im, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, src_psi_B2_re, src_psi_B2_im, t, x2, x1, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f);
               make_single_block(B2_Bsingle_r2_re, B2_Bsingle_r2_im, B2_Blocal_r2_re, B2_Blocal_r2_im, B2_prop_re, B2_prop_im, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, src_psi_B2_re, src_psi_B2_im, t, x2, x1, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f);
               make_double_block(B2_Bdouble_r1_re, B2_Bdouble_r1_im, B2_prop_re, B2_prop_im, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, src_psi_B2_re, src_psi_B2_im, t, x2, x1, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f);
               make_double_block(B2_Bdouble_r2_re, B2_Bdouble_r2_im, B2_prop_re, B2_prop_im, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, src_psi_B2_re, src_psi_B2_im, t, x2, x1, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f);
               /* compute two nucleon correlators from blocks */
               make_dibaryon_correlator(BB_0_re, BB_0_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B1_Bsingle_r1_re, B1_Bsingle_r1_im, B1_Bdouble_r1_re, B1_Bdouble_r1_im, B2_Blocal_r2_re, B2_Blocal_r2_im, B2_Bsingle_r2_re, B2_Bsingle_r2_im, B2_Bdouble_r2_re, B2_Bdouble_r2_im, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_1, snk_spin_weights_1, snk_weights_1, snk_psi_re, snk_psi_im, t, x1, x2, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f,Nsnk_f,Nperms_f);
               make_dibaryon_correlator(BB_0_re, BB_0_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B1_Bsingle_r1_re, B1_Bsingle_r1_im, B1_Bdouble_r1_re, B1_Bdouble_r1_im, B2_Blocal_r2_re, B2_Blocal_r2_im, B2_Bsingle_r2_re, B2_Bsingle_r2_im, B2_Bdouble_r2_re, B2_Bdouble_r2_im, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_2, snk_spin_weights_2, snk_weights_2, snk_psi_re, snk_psi_im, t, x1, x2, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f,Nsnk_f,Nperms_f);
               make_dibaryon_correlator(BB_0_re, BB_0_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B1_Bsingle_r2_re, B1_Bsingle_r2_im, B1_Bdouble_r2_re, B1_Bdouble_r2_im, B2_Blocal_r1_re, B2_Blocal_r1_im, B2_Bsingle_r1_re, B2_Bsingle_r1_im, B2_Bdouble_r1_re, B2_Bdouble_r1_im, perms, sigs, -1.0*overall_weight/sqrt(2.0), snk_color_weights_1, snk_spin_weights_1, snk_weights_1, snk_psi_re, snk_psi_im, t, x1, x2, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f,Nsnk_f,Nperms_f);
               make_dibaryon_correlator(BB_0_re, BB_0_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B1_Bsingle_r2_re, B1_Bsingle_r2_im, B1_Bdouble_r2_re, B1_Bdouble_r2_im, B2_Blocal_r1_re, B2_Blocal_r1_im, B2_Bsingle_r1_re, B2_Bsingle_r1_im, B2_Bdouble_r1_re, B2_Bdouble_r1_im, perms, sigs, -1.0*overall_weight/sqrt(2.0), snk_color_weights_2, snk_spin_weights_2, snk_weights_2, snk_psi_re, snk_psi_im, t, x1, x2, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f,Nsnk_f,Nperms_f);
               //make_dibaryon_correlator(BB_r1_re, BB_r1_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B1_Bsingle_r1_re, B1_Bsingle_r1_im, B1_Bdouble_r1_re, B1_Bdouble_r1_im, B2_Blocal_r1_re, B2_Blocal_r1_im, B2_Bsingle_r1_re, B2_Bsingle_r1_im, B2_Bdouble_r1_re, B2_Bdouble_r1_im, perms, sigs, overall_weight, snk_color_weights_r1, snk_spin_weights_r1, snk_weights_r1, snk_psi_re, snk_psi_im, t, x1, x2, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f,Nsnk_f,Nperms_f);
               //make_dibaryon_correlator(BB_r2_re, BB_r2_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B1_Bsingle_r1_re, B1_Bsingle_r1_im, B1_Bdouble_r1_re, B1_Bdouble_r1_im, B2_Blocal_r2_re, B2_Blocal_r2_im, B2_Bsingle_r2_re, B2_Bsingle_r2_im, B2_Bdouble_r2_re, B2_Bdouble_r2_im, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_r2_1, snk_spin_weights_r2_1, snk_weights_r2_1, snk_psi_re, snk_psi_im, t, x1, x2,  Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f,Nsnk_f,Nperms_f);
               //make_dibaryon_correlator(BB_r2_re, BB_r2_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B1_Bsingle_r1_re, B1_Bsingle_r1_im, B1_Bdouble_r1_re, B1_Bdouble_r1_im, B2_Blocal_r2_re, B2_Blocal_r2_im, B2_Bsingle_r2_re, B2_Bsingle_r2_im, B2_Bdouble_r2_re, B2_Bdouble_r2_im, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_r2_2, snk_spin_weights_r2_2, snk_weights_r2_2, snk_psi_re, snk_psi_im, t, x1, x2, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f,Nsnk_f,Nperms_f);
               //make_dibaryon_correlator(BB_r2_re, BB_r2_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B1_Bsingle_r2_re, B1_Bsingle_r2_im, B1_Bdouble_r2_re, B1_Bdouble_r2_im, B2_Blocal_r1_re, B2_Blocal_r1_im, B2_Bsingle_r1_re, B2_Bsingle_r1_im, B2_Bdouble_r1_re, B2_Bdouble_r1_im, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_r2_1, snk_spin_weights_r2_1, snk_weights_r2_1, snk_psi_re, snk_psi_im, t, x1, x2, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f,Nsnk_f,Nperms_f);
               //make_dibaryon_correlator(BB_r2_re, BB_r2_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B1_Bsingle_r2_re, B1_Bsingle_r2_im, B1_Bdouble_r2_re, B1_Bdouble_r2_im, B2_Blocal_r1_re, B2_Blocal_r1_im, B2_Bsingle_r1_re, B2_Bsingle_r1_im, B2_Bdouble_r1_re, B2_Bdouble_r1_im, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_r2_2, snk_spin_weights_r2_2, snk_weights_r2_2, snk_psi_re, snk_psi_im, t, x1, x2, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f,Nsnk_f,Nperms_f);
               //make_dibaryon_correlator(BB_r3_re, BB_r3_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B1_Bsingle_r2_re, B1_Bsingle_r2_im, B1_Bdouble_r2_re, B1_Bdouble_r2_im, B2_Blocal_r2_re, B2_Blocal_r2_im, B2_Bsingle_r2_re, B2_Bsingle_r2_im, B2_Bdouble_r2_re, B2_Bdouble_r2_im, perms, sigs, overall_weight, snk_color_weights_r3, snk_spin_weights_r3, snk_weights_r3, snk_psi_re, snk_psi_im, t, x1, x2, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f,Nsnk_f,Nperms_f);
               if (space_symmetric != 0) {
                  make_dibaryon_correlator(BB_0_re, BB_0_im, B2_Blocal_r1_re, B2_Blocal_r1_im, B2_Bsingle_r1_re, B2_Bsingle_r1_im, B2_Bdouble_r1_re, B2_Bdouble_r1_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B1_Bsingle_r2_re, B1_Bsingle_r2_im, B1_Bdouble_r2_re, B1_Bdouble_r2_im, perms, sigs, space_symmetric*overall_weight/sqrt(2.0), snk_color_weights_1, snk_spin_weights_1, snk_weights_1, snk_psi_re, snk_psi_im, t, x1, x2, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f,Nsnk_f,Nperms_f);
                  make_dibaryon_correlator(BB_0_re, BB_0_im, B2_Blocal_r1_re, B2_Blocal_r1_im, B2_Bsingle_r1_re, B2_Bsingle_r1_im, B2_Bdouble_r1_re, B2_Bdouble_r1_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B1_Bsingle_r2_re, B1_Bsingle_r2_im, B1_Bdouble_r2_re, B1_Bdouble_r2_im, perms, sigs, space_symmetric*overall_weight/sqrt(2.0), snk_color_weights_2, snk_spin_weights_2, snk_weights_2, snk_psi_re, snk_psi_im, t, x1, x2, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f,Nsnk_f,Nperms_f);
                  make_dibaryon_correlator(BB_0_re, BB_0_im, B2_Blocal_r2_re, B2_Blocal_r2_im, B2_Bsingle_r2_re, B2_Bsingle_r2_im, B2_Bdouble_r2_re, B2_Bdouble_r2_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B1_Bsingle_r1_re, B1_Bsingle_r1_im, B1_Bdouble_r1_re, B1_Bdouble_r1_im, perms, sigs, -1.0*space_symmetric*overall_weight/sqrt(2.0), snk_color_weights_1, snk_spin_weights_1, snk_weights_1, snk_psi_re, snk_psi_im, t, x1, x2, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f,Nsnk_f,Nperms_f);
                  make_dibaryon_correlator(BB_0_re, BB_0_im, B2_Blocal_r2_re, B2_Blocal_r2_im, B2_Bsingle_r2_re, B2_Bsingle_r2_im, B2_Bdouble_r2_re, B2_Bdouble_r2_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B1_Bsingle_r1_re, B1_Bsingle_r1_im, B1_Bdouble_r1_re, B1_Bdouble_r1_im, perms, sigs, -1.0*space_symmetric*overall_weight/sqrt(2.0), snk_color_weights_2, snk_spin_weights_2, snk_weights_2, snk_psi_re, snk_psi_im, t, x1, x2, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f,Nsnk_f,Nperms_f);
                  make_dibaryon_correlator(BB_r1_re, BB_r1_im, B2_Blocal_r1_re, B2_Blocal_r1_im, B2_Bsingle_r1_re, B2_Bsingle_r1_im, B2_Bdouble_r1_re, B2_Bdouble_r1_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B1_Bsingle_r1_re, B1_Bsingle_r1_im, B1_Bdouble_r1_re, B1_Bdouble_r1_im, perms, sigs, space_symmetric*overall_weight, snk_color_weights_r1, snk_spin_weights_r1, snk_weights_r1, snk_psi_re, snk_psi_im, t, x1, x2, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f,Nsnk_f,Nperms_f);
                  make_dibaryon_correlator(BB_r2_re, BB_r2_im, B2_Blocal_r1_re, B2_Blocal_r1_im, B2_Bsingle_r1_re, B2_Bsingle_r1_im, B2_Bdouble_r1_re, B2_Bdouble_r1_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B1_Bsingle_r2_re, B1_Bsingle_r2_im, B1_Bdouble_r2_re, B1_Bdouble_r2_im, perms, sigs, space_symmetric*overall_weight/sqrt(2.0), snk_color_weights_r2_1, snk_spin_weights_r2_1, snk_weights_r2_1, snk_psi_re, snk_psi_im, t, x1, x2, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f,Nsnk_f,Nperms_f);
                  make_dibaryon_correlator(BB_r2_re, BB_r2_im, B2_Blocal_r1_re, B2_Blocal_r1_im, B2_Bsingle_r1_re, B2_Bsingle_r1_im, B2_Bdouble_r1_re, B2_Bdouble_r1_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B1_Bsingle_r2_re, B1_Bsingle_r2_im, B1_Bdouble_r2_re, B1_Bdouble_r2_im, perms, sigs, space_symmetric*overall_weight/sqrt(2.0), snk_color_weights_r2_2, snk_spin_weights_r2_2, snk_weights_r2_2, snk_psi_re, snk_psi_im, t, x1, x2, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f,Nsnk_f,Nperms_f);
                  make_dibaryon_correlator(BB_r2_re, BB_r2_im, B2_Blocal_r2_re, B2_Blocal_r2_im, B2_Bsingle_r2_re, B2_Bsingle_r2_im, B2_Bdouble_r2_re, B2_Bdouble_r2_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B1_Bsingle_r1_re, B1_Bsingle_r1_im, B1_Bdouble_r1_re, B1_Bdouble_r1_im, perms, sigs, space_symmetric*overall_weight/sqrt(2.0), snk_color_weights_r2_1, snk_spin_weights_r2_1, snk_weights_r2_1, snk_psi_re, snk_psi_im, t, x1, x2, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f,Nsnk_f,Nperms_f);
                  make_dibaryon_correlator(BB_r2_re, BB_r2_im, B2_Blocal_r2_re, B2_Blocal_r2_im, B2_Bsingle_r2_re, B2_Bsingle_r2_im, B2_Bdouble_r2_re, B2_Bdouble_r2_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B1_Bsingle_r1_re, B1_Bsingle_r1_im, B1_Bdouble_r1_re, B1_Bdouble_r1_im, perms, sigs, space_symmetric*overall_weight/sqrt(2.0), snk_color_weights_r2_2, snk_spin_weights_r2_2, snk_weights_r2_2, snk_psi_re, snk_psi_im, t, x1, x2, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f,Nsnk_f,Nperms_f);
                  make_dibaryon_correlator(BB_r3_re, BB_r3_im, B2_Blocal_r2_re, B2_Blocal_r2_im, B2_Bsingle_r2_re, B2_Bsingle_r2_im, B2_Bdouble_r2_re, B2_Bdouble_r2_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B1_Bsingle_r2_re, B1_Bsingle_r2_im, B1_Bdouble_r2_re, B1_Bdouble_r2_im, perms, sigs, space_symmetric*overall_weight, snk_color_weights_r3, snk_spin_weights_r3, snk_weights_r3, snk_psi_re, snk_psi_im, t, x1, x2, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f,Nsnk_f,Nperms_f);
               }
            }
         }
         for (m=0; m<Nsrc_f; m++) {
            for (n=0; n<Nsnk_f; n++) {
               C_re[index_4d(0,m,n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] = BB_0_re[index_3d(m,n,t,Nsnk_f,Nt_f)];
               C_im[index_4d(0,m,n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] = BB_0_im[index_3d(m,n,t,Nsnk_f,Nt_f)];
               C_re[index_4d(1,m,n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] = BB_r1_re[index_3d(m,n,t,Nsnk_f,Nt_f)];
               C_im[index_4d(1,m,n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] = BB_r1_im[index_3d(m,n,t,Nsnk_f,Nt_f)];
               C_re[index_4d(2,m,n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] = BB_r2_re[index_3d(m,n,t,Nsnk_f,Nt_f)];
               C_im[index_4d(2,m,n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] = BB_r2_im[index_3d(m,n,t,Nsnk_f,Nt_f)];
               C_re[index_4d(3,m,n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] = BB_r3_re[index_3d(m,n,t,Nsnk_f,Nt_f)];
               C_im[index_4d(3,m,n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] = BB_r3_im[index_3d(m,n,t,Nsnk_f,Nt_f)];
            }
         }
      }
      free(B1_Blocal_r1_re);
      free(B1_Blocal_r1_im);
      free(B1_Blocal_r2_re);
      free(B1_Blocal_r2_im);
      free(B2_Blocal_r1_re);
      free(B2_Blocal_r1_im);
      free(B2_Blocal_r2_re);
      free(B2_Blocal_r2_im);
      free(B1_Bsingle_r1_re);
      free(B1_Bsingle_r1_im);
      free(B1_Bsingle_r2_re);
      free(B1_Bsingle_r2_im);
      free(B1_Bdouble_r1_re);
      free(B1_Bdouble_r1_im);
      free(B1_Bdouble_r2_re);
      free(B1_Bdouble_r2_im);
      free(B2_Bsingle_r1_re);
      free(B2_Bsingle_r1_im);
      free(B2_Bsingle_r2_re);
      free(B2_Bsingle_r2_im);
      free(B2_Bdouble_r1_re);
      free(B2_Bdouble_r1_im);
      free(B2_Bdouble_r2_re);
      free(B2_Bdouble_r2_im);
      printf("made BB-BB \n");
   }
   free(BB_0_re);
   free(BB_0_im);
   free(BB_r1_re);
   free(BB_r1_im);
   free(BB_r2_re);
   free(BB_r2_im);
   free(BB_r3_re);
   free(BB_r3_im);
   if (Nsrc_f > 0 && Nsnk_fHex > 0) {
      // BB_H 
      double* B1_Blocal_r1_re = (double *) malloc(Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsrc_f * sizeof (double));
      double* B1_Blocal_r1_im = (double *) malloc(Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsrc_f * sizeof (double));
      double* B1_Blocal_r2_re = (double *) malloc(Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsrc_f * sizeof (double));
      double* B1_Blocal_r2_im = (double *) malloc(Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsrc_f * sizeof (double));
      double* B2_Blocal_r1_re = (double *) malloc(Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsrc_f * sizeof (double));
      double* B2_Blocal_r1_im = (double *) malloc(Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsrc_f * sizeof (double));
      double* B2_Blocal_r2_re = (double *) malloc(Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsrc_f * sizeof (double));
      double* B2_Blocal_r2_im = (double *) malloc(Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsrc_f * sizeof (double));
      for (t=0; t<Nt_f; t++) {
         for (x =0; x<Vsnk_f; x++) {
            // create blocks
            make_local_block(B1_Blocal_r1_re, B1_Blocal_r1_im, B1_prop_re, B1_prop_im, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, src_psi_B1_re, src_psi_B1_im, t, x, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f);
            make_local_block(B1_Blocal_r2_re, B1_Blocal_r2_im, B1_prop_re, B1_prop_im, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, src_psi_B1_re, src_psi_B1_im, t, x, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f);
            make_local_block(B2_Blocal_r1_re, B2_Blocal_r1_im, B2_prop_re, B2_prop_im, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, src_psi_B2_re, src_psi_B2_im, t, x, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f);
            make_local_block(B2_Blocal_r2_re, B2_Blocal_r2_im, B2_prop_re, B2_prop_im, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, src_psi_B2_re, src_psi_B2_im, t, x, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f);
            make_dibaryon_hex_correlator(BB_H_0_re, BB_H_0_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B2_Blocal_r2_re, B2_Blocal_r2_im, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_1, snk_spin_weights_1, snk_weights_1, hex_snk_psi_re, hex_snk_psi_im, t, x, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f,Nsnk_fHex,Nperms_f);
            make_dibaryon_hex_correlator(BB_H_0_re, BB_H_0_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B2_Blocal_r2_re, B2_Blocal_r2_im, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_2, snk_spin_weights_2, snk_weights_2, hex_snk_psi_re, hex_snk_psi_im, t, x, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f,Nsnk_fHex,Nperms_f);
            make_dibaryon_hex_correlator(BB_H_0_re, BB_H_0_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B2_Blocal_r1_re, B2_Blocal_r1_im, perms, sigs, -1.0*overall_weight/sqrt(2.0), snk_color_weights_1, snk_spin_weights_1, snk_weights_1, hex_snk_psi_re, hex_snk_psi_im, t, x, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f,Nsnk_fHex,Nperms_f);
            make_dibaryon_hex_correlator(BB_H_0_re, BB_H_0_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B2_Blocal_r1_re, B2_Blocal_r1_im, perms, sigs, -1.0*overall_weight/sqrt(2.0), snk_color_weights_2, snk_spin_weights_2, snk_weights_2, hex_snk_psi_re, hex_snk_psi_im, t, x, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f,Nsnk_fHex,Nperms_f);
            make_dibaryon_hex_correlator(BB_H_r1_re, BB_H_r1_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B2_Blocal_r1_re, B2_Blocal_r1_im, perms, sigs, overall_weight, snk_color_weights_r1, snk_spin_weights_r1, snk_weights_r1, hex_snk_psi_re, hex_snk_psi_im, t, x, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f,Nsnk_fHex,Nperms_f);
            make_dibaryon_hex_correlator(BB_H_r2_re, BB_H_r2_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B2_Blocal_r2_re, B2_Blocal_r2_im, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_r2_1, snk_spin_weights_r2_1, snk_weights_r2_1, hex_snk_psi_re, hex_snk_psi_im, t, x, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f,Nsnk_fHex,Nperms_f);
            make_dibaryon_hex_correlator(BB_H_r2_re, BB_H_r2_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B2_Blocal_r2_re, B2_Blocal_r2_im, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_r2_2, snk_spin_weights_r2_2, snk_weights_r2_2, hex_snk_psi_re, hex_snk_psi_im, t, x, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f,Nsnk_fHex,Nperms_f);
            make_dibaryon_hex_correlator(BB_H_r2_re, BB_H_r2_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B2_Blocal_r1_re, B2_Blocal_r1_im, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_r2_1, snk_spin_weights_r2_1, snk_weights_r2_1, hex_snk_psi_re, hex_snk_psi_im, t, x, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f,Nsnk_fHex,Nperms_f);
            make_dibaryon_hex_correlator(BB_H_r2_re, BB_H_r2_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B2_Blocal_r1_re, B2_Blocal_r1_im, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_r2_2, snk_spin_weights_r2_2, snk_weights_r2_2, hex_snk_psi_re, hex_snk_psi_im, t, x, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f,Nsnk_fHex,Nperms_f);
            make_dibaryon_hex_correlator(BB_H_r3_re, BB_H_r3_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B2_Blocal_r2_re, B2_Blocal_r2_im, perms, sigs, overall_weight, snk_color_weights_r3, snk_spin_weights_r3, snk_weights_r3, hex_snk_psi_re, hex_snk_psi_im, t, x, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f,Nsnk_fHex,Nperms_f);
            if (space_symmetric != 0) {
               make_dibaryon_hex_correlator(BB_H_0_re, BB_H_0_im, B2_Blocal_r1_re, B2_Blocal_r1_im, B1_Blocal_r2_re, B1_Blocal_r2_im, perms, sigs, space_symmetric*overall_weight/sqrt(2.0), snk_color_weights_1, snk_spin_weights_1, snk_weights_1, hex_snk_psi_re, hex_snk_psi_im, t, x, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f,Nsnk_fHex,Nperms_f);
               make_dibaryon_hex_correlator(BB_H_0_re, BB_H_0_im, B2_Blocal_r1_re, B2_Blocal_r1_im, B1_Blocal_r2_re, B1_Blocal_r2_im, perms, sigs, space_symmetric*overall_weight/sqrt(2.0), snk_color_weights_2, snk_spin_weights_2, snk_weights_2, hex_snk_psi_re, hex_snk_psi_im, t, x, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f,Nsnk_fHex,Nperms_f);
               make_dibaryon_hex_correlator(BB_H_0_re, BB_H_0_im, B2_Blocal_r2_re, B2_Blocal_r2_im, B1_Blocal_r1_re, B1_Blocal_r1_im, perms, sigs, -1.0*space_symmetric*overall_weight/sqrt(2.0), snk_color_weights_1, snk_spin_weights_1, snk_weights_1, hex_snk_psi_re, hex_snk_psi_im, t, x, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f,Nsnk_fHex,Nperms_f);
               make_dibaryon_hex_correlator(BB_H_0_re, BB_H_0_im, B2_Blocal_r2_re, B2_Blocal_r2_im, B1_Blocal_r1_re, B1_Blocal_r1_im, perms, sigs, -1.0*space_symmetric*overall_weight/sqrt(2.0), snk_color_weights_2, snk_spin_weights_2, snk_weights_2, hex_snk_psi_re, hex_snk_psi_im, t, x, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f,Nsnk_fHex,Nperms_f);
               make_dibaryon_hex_correlator(BB_H_r1_re, BB_H_r1_im, B2_Blocal_r1_re, B2_Blocal_r1_im, B1_Blocal_r1_re, B1_Blocal_r1_im, perms, sigs, space_symmetric*overall_weight, snk_color_weights_r1, snk_spin_weights_r1, snk_weights_r1, hex_snk_psi_re, hex_snk_psi_im, t, x, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f,Nsnk_fHex,Nperms_f);
               make_dibaryon_hex_correlator(BB_H_r2_re, BB_H_r2_im, B2_Blocal_r1_re, B2_Blocal_r1_im, B1_Blocal_r2_re, B1_Blocal_r2_im, perms, sigs, space_symmetric*overall_weight/sqrt(2.0), snk_color_weights_r2_1, snk_spin_weights_r2_1, snk_weights_r2_1, hex_snk_psi_re, hex_snk_psi_im, t, x, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f,Nsnk_fHex,Nperms_f);
               make_dibaryon_hex_correlator(BB_H_r2_re, BB_H_r2_im, B2_Blocal_r1_re, B2_Blocal_r1_im, B1_Blocal_r2_re, B1_Blocal_r2_im, perms, sigs, space_symmetric*overall_weight/sqrt(2.0), snk_color_weights_r2_2, snk_spin_weights_r2_2, snk_weights_r2_2, hex_snk_psi_re, hex_snk_psi_im, t, x, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f,Nsnk_fHex,Nperms_f);
               make_dibaryon_hex_correlator(BB_H_r2_re, BB_H_r2_im, B2_Blocal_r2_re, B2_Blocal_r2_im, B1_Blocal_r1_re, B1_Blocal_r1_im, perms, sigs, space_symmetric*overall_weight/sqrt(2.0), snk_color_weights_r2_1, snk_spin_weights_r2_1, snk_weights_r2_1, hex_snk_psi_re, hex_snk_psi_im, t, x, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f,Nsnk_fHex,Nperms_f);
               make_dibaryon_hex_correlator(BB_H_r2_re, BB_H_r2_im, B2_Blocal_r2_re, B2_Blocal_r2_im, B1_Blocal_r1_re, B1_Blocal_r1_im, perms, sigs, space_symmetric*overall_weight/sqrt(2.0), snk_color_weights_r2_2, snk_spin_weights_r2_2, snk_weights_r2_2, hex_snk_psi_re, hex_snk_psi_im, t, x, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f,Nsnk_fHex,Nperms_f);
               make_dibaryon_hex_correlator(BB_H_r3_re, BB_H_r3_im, B2_Blocal_r2_re, B2_Blocal_r2_im, B1_Blocal_r2_re, B1_Blocal_r2_im, perms, sigs, space_symmetric*overall_weight, snk_color_weights_r3, snk_spin_weights_r3, snk_weights_r3, hex_snk_psi_re, hex_snk_psi_im, t, x, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_f,Nsnk_fHex,Nperms_f);
            }
         }
         for (m=0; m<Nsrc_f; m++) {
            for (n=0; n<Nsnk_fHex; n++) {
               C_re[index_4d(0,m,Nsnk_f+n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] = BB_H_0_re[index_3d(m,n,t ,Nsnk_fHex,Nt_f)];
               C_im[index_4d(0,m,Nsnk_f+n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] = BB_H_0_im[index_3d(m,n,t ,Nsnk_fHex,Nt_f)];
               C_re[index_4d(1,m,Nsnk_f+n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] = BB_H_r1_re[index_3d(m,n,t ,Nsnk_fHex,Nt_f)];
               C_im[index_4d(1,m,Nsnk_f+n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] = BB_H_r1_im[index_3d(m,n,t ,Nsnk_fHex,Nt_f)];
               C_re[index_4d(2,m,Nsnk_f+n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] = BB_H_r2_re[index_3d(m,n,t ,Nsnk_fHex,Nt_f)];
               C_im[index_4d(2,m,Nsnk_f+n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] = BB_H_r2_im[index_3d(m,n,t ,Nsnk_fHex,Nt_f)];
               C_re[index_4d(3,m,Nsnk_f+n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] = BB_H_r3_re[index_3d(m,n,t ,Nsnk_fHex,Nt_f)];
               C_im[index_4d(3,m,Nsnk_f+n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] = BB_H_r3_im[index_3d(m,n,t ,Nsnk_fHex,Nt_f)];
            }
         } 
         if (snk_entangled != 0 && Nsrc_f == Nsnk_f && Nsrc_fHex == Nsnk_fHex) {
            for (m=0; m<Nsrc_fHex; m++) {
               for (n=0; n<Nsnk_f; n++) {
                  C_re[index_4d(0,Nsrc_f+m,n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] = BB_H_0_re[index_3d(m,n,t,Nsnk_fHex,Nt_f)];
                  C_im[index_4d(0,Nsrc_f+m,n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] = -BB_H_0_im[index_3d(m,n,t,Nsnk_fHex,Nt_f)];
                  C_re[index_4d(1,Nsrc_f+m,n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] = BB_H_r1_re[index_3d(m,n,t,Nsnk_fHex,Nt_f)];
                  C_im[index_4d(1,Nsrc_f+m,n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] = -BB_H_r1_im[index_3d(m,n,t,Nsnk_fHex,Nt_f)];
                  C_re[index_4d(2,Nsrc_f+m,n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] = BB_H_r2_re[index_3d(m,n,t,Nsnk_fHex,Nt_f)];
                  C_im[index_4d(2,Nsrc_f+m,n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] = -BB_H_r2_im[index_3d(m,n,t,Nsnk_fHex,Nt_f)];
                  C_re[index_4d(3,Nsrc_f+m,n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] = BB_H_r3_re[index_3d(m,n,t,Nsnk_fHex,Nt_f)];
                  C_im[index_4d(3,Nsrc_f+m,n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] = -BB_H_r3_im[index_3d(m,n,t,Nsnk_fHex,Nt_f)];
               }
            }
         } 
      }
      free(B1_Blocal_r1_re);
      free(B1_Blocal_r1_im);
      free(B1_Blocal_r2_re);
      free(B1_Blocal_r2_im);
      free(B2_Blocal_r1_re);
      free(B2_Blocal_r1_im);
      free(B2_Blocal_r2_re);
      free(B2_Blocal_r2_im);
      printf("made BB-H \n");
   }
   free(BB_H_0_re);
   free(BB_H_0_im);
   free(BB_H_r1_re);
   free(BB_H_r1_im);
   free(BB_H_r2_re);
   free(BB_H_r2_im);
   free(BB_H_r3_re);
   free(BB_H_r3_im);
   if (Nsrc_fHex > 0 && Nsnk_fHex > 0) {
      // H_H 
      make_hex_correlator(H_0_re, H_0_im, B1_prop_re, B1_prop_im, B2_prop_re, B2_prop_im, perms, sigs, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, overall_weight/sqrt(2.0), snk_color_weights_1, snk_spin_weights_1, snk_weights_1, hex_src_psi_re, hex_src_psi_im, hex_snk_psi_re, hex_snk_psi_im, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_fHex,Nsnk_fHex,Nperms_f);
      make_hex_correlator(H_0_re, H_0_im, B1_prop_re, B1_prop_im, B2_prop_re, B2_prop_im, perms, sigs, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, -1.0*overall_weight/sqrt(2.0), snk_color_weights_1, snk_spin_weights_1, snk_weights_1, hex_src_psi_re, hex_src_psi_im, hex_snk_psi_re, hex_snk_psi_im, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_fHex,Nsnk_fHex,Nperms_f);
      make_hex_correlator(H_0_re, H_0_im, B1_prop_re, B1_prop_im, B2_prop_re, B2_prop_im, perms, sigs, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, overall_weight/sqrt(2.0), snk_color_weights_2, snk_spin_weights_2, snk_weights_2, hex_src_psi_re, hex_src_psi_im, hex_snk_psi_re, hex_snk_psi_im, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_fHex,Nsnk_fHex,Nperms_f);
      make_hex_correlator(H_0_re, H_0_im, B1_prop_re, B1_prop_im, B2_prop_re, B2_prop_im, perms, sigs, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, -1.0*overall_weight/sqrt(2.0), snk_color_weights_2, snk_spin_weights_2, snk_weights_2, hex_src_psi_re, hex_src_psi_im, hex_snk_psi_re, hex_snk_psi_im, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_fHex,Nsnk_fHex,Nperms_f);
      make_hex_correlator(H_r1_re, H_r1_im, B1_prop_re, B1_prop_im, B2_prop_re, B2_prop_im, perms, sigs, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, overall_weight, snk_color_weights_r1, snk_spin_weights_r1, snk_weights_r1, hex_src_psi_re, hex_src_psi_im, hex_snk_psi_re, hex_snk_psi_im, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_fHex,Nsnk_fHex,Nperms_f);
      make_hex_correlator(H_r2_re, H_r2_im, B1_prop_re, B1_prop_im, B2_prop_re, B2_prop_im, perms, sigs, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, overall_weight/sqrt(2.0), snk_color_weights_r2_1, snk_spin_weights_r2_1, snk_weights_r2_1, hex_src_psi_re, hex_src_psi_im, hex_snk_psi_re, hex_snk_psi_im, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_fHex,Nsnk_fHex,Nperms_f);
      make_hex_correlator(H_r2_re, H_r2_im, B1_prop_re, B1_prop_im, B2_prop_re, B2_prop_im, perms, sigs, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, overall_weight/sqrt(2.0), snk_color_weights_r2_1, snk_spin_weights_r2_1, snk_weights_r2_1, hex_src_psi_re, hex_src_psi_im, hex_snk_psi_re, hex_snk_psi_im, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_fHex,Nsnk_fHex,Nperms_f);
      make_hex_correlator(H_r2_re, H_r2_im, B1_prop_re, B1_prop_im, B2_prop_re, B2_prop_im, perms, sigs, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, overall_weight/sqrt(2.0), snk_color_weights_r2_2, snk_spin_weights_r2_2, snk_weights_r2_2, hex_src_psi_re, hex_src_psi_im, hex_snk_psi_re, hex_snk_psi_im, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_fHex,Nsnk_fHex,Nperms_f);
      make_hex_correlator(H_r2_re, H_r2_im, B1_prop_re, B1_prop_im, B2_prop_re, B2_prop_im, perms, sigs, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, overall_weight/sqrt(2.0), snk_color_weights_r2_2, snk_spin_weights_r2_2, snk_weights_r2_2, hex_src_psi_re, hex_src_psi_im, hex_snk_psi_re, hex_snk_psi_im, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_fHex,Nsnk_fHex,Nperms_f);
      make_hex_correlator(H_r3_re, H_r3_im, B1_prop_re, B1_prop_im, B2_prop_re, B2_prop_im, perms, sigs, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, overall_weight, snk_color_weights_r3, snk_spin_weights_r3, snk_weights_r3, hex_src_psi_re, hex_src_psi_im, hex_snk_psi_re, hex_snk_psi_im, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_fHex,Nsnk_fHex,Nperms_f);
      for (m=0; m<Nsrc_fHex; m++) {
         for (n=0; n<Nsnk_fHex; n++) {
            for (t=0; t<Nt_f; t++) {
               C_re[index_4d(0,Nsrc_f+m,Nsnk_f+n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] = H_0_re[index_3d(m,n,t ,Nsnk_fHex,Nt_f)];
               C_im[index_4d(0,Nsrc_f+m,Nsnk_f+n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] = H_0_im[index_3d(m,n,t ,Nsnk_fHex,Nt_f)];
               C_re[index_4d(1,Nsrc_f+m,Nsnk_f+n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] = H_r1_re[index_3d(m,n,t ,Nsnk_fHex,Nt_f)];
               C_im[index_4d(1,Nsrc_f+m,Nsnk_f+n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] = H_r1_im[index_3d(m,n,t ,Nsnk_fHex,Nt_f)];
               C_re[index_4d(2,Nsrc_f+m,Nsnk_f+n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] = H_r2_re[index_3d(m,n,t ,Nsnk_fHex,Nt_f)];
               C_im[index_4d(2,Nsrc_f+m,Nsnk_f+n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] = H_r2_im[index_3d(m,n,t ,Nsnk_fHex,Nt_f)];
               C_re[index_4d(3,Nsrc_f+m,Nsnk_f+n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] = H_r3_re[index_3d(m,n,t ,Nsnk_fHex,Nt_f)];
               C_im[index_4d(3,Nsrc_f+m,Nsnk_f+n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] = H_r3_im[index_3d(m,n,t ,Nsnk_fHex,Nt_f)];
            }
         }
      }
      printf("made H-H \n");
   }
   free(H_0_re);
   free(H_0_im);
   free(H_r1_re);
   free(H_r1_im);
   free(H_r2_re);
   free(H_r2_im);
   free(H_r3_re);
   free(H_r3_im);
   if (Nsnk_f > 0 && Nsrc_fHex > 0 && snk_entangled == 0) {
      // H_BB 
      double* snk_B1_Blocal_r1_re = (double *) malloc(Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsnk_f * sizeof (double));
      double* snk_B1_Blocal_r1_im = (double *) malloc(Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsnk_f * sizeof (double));
      double* snk_B1_Blocal_r2_re = (double *) malloc(Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsnk_f * sizeof (double));
      double* snk_B1_Blocal_r2_im = (double *) malloc(Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsnk_f * sizeof (double));
      double* snk_B2_Blocal_r1_re = (double *) malloc(Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsnk_f * sizeof (double));
      double* snk_B2_Blocal_r1_im = (double *) malloc(Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsnk_f * sizeof (double));
      double* snk_B2_Blocal_r2_re = (double *) malloc(Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsnk_f * sizeof (double));
      double* snk_B2_Blocal_r2_im = (double *) malloc(Nc_f * Ns_f * Nc_f * Ns_f * Nc_f * Ns_f * Nsnk_f * sizeof (double));
      for (t=0; t<Nt_f; t++) {
         for (y =0; y<Vsrc_f; y++) {
            // create blocks
            make_local_snk_block(snk_B1_Blocal_r1_re, snk_B1_Blocal_r1_im, B1_prop_re, B1_prop_im, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, snk_psi_B1_re, snk_psi_B1_im, t, y, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsnk_f);
            make_local_snk_block(snk_B1_Blocal_r2_re, snk_B1_Blocal_r2_im, B1_prop_re, B1_prop_im, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, snk_psi_B1_re, snk_psi_B1_im, t, y, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsnk_f);
            make_local_snk_block(snk_B2_Blocal_r1_re, snk_B2_Blocal_r1_im, B2_prop_re, B2_prop_im, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, snk_psi_B2_re, snk_psi_B2_im, t, y, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsnk_f);
            make_local_snk_block(snk_B2_Blocal_r2_re, snk_B2_Blocal_r2_im, B2_prop_re, B2_prop_im, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, snk_psi_B2_re, snk_psi_B2_im, t, y, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsnk_f);
            make_hex_dibaryon_correlator(H_BB_0_re, H_BB_0_im, snk_B1_Blocal_r1_re, snk_B1_Blocal_r1_im, snk_B2_Blocal_r2_re, snk_B2_Blocal_r2_im, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_1, snk_spin_weights_1, snk_weights_1, hex_src_psi_re, hex_src_psi_im, t, y, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_fHex,Nsnk_f,Nperms_f);
            make_hex_dibaryon_correlator(H_BB_0_re, H_BB_0_im, snk_B1_Blocal_r1_re, snk_B1_Blocal_r1_im, snk_B2_Blocal_r2_re, snk_B2_Blocal_r2_im, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_2, snk_spin_weights_2, snk_weights_2, hex_src_psi_re, hex_src_psi_im, t, y, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_fHex,Nsnk_f,Nperms_f);
            make_hex_dibaryon_correlator(H_BB_0_re, H_BB_0_im, snk_B1_Blocal_r2_re, snk_B1_Blocal_r2_im, snk_B2_Blocal_r1_re, snk_B2_Blocal_r1_im, perms, sigs, -1.0*overall_weight/sqrt(2.0), snk_color_weights_1, snk_spin_weights_1, snk_weights_1, hex_src_psi_re, hex_src_psi_im, t, y, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_fHex,Nsnk_f,Nperms_f);
            make_hex_dibaryon_correlator(H_BB_0_re, H_BB_0_im, snk_B1_Blocal_r2_re, snk_B1_Blocal_r2_im, snk_B2_Blocal_r1_re, snk_B2_Blocal_r1_im, perms, sigs, -1.0*overall_weight/sqrt(2.0), snk_color_weights_2, snk_spin_weights_2, snk_weights_2, hex_src_psi_re, hex_src_psi_im, t, y, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_fHex,Nsnk_f,Nperms_f);
            make_hex_dibaryon_correlator(H_BB_r1_re, H_BB_r1_im, snk_B1_Blocal_r1_re, snk_B1_Blocal_r1_im, snk_B2_Blocal_r1_re, snk_B2_Blocal_r1_im, perms, sigs, overall_weight, snk_color_weights_r1, snk_spin_weights_r1, snk_weights_r1, hex_src_psi_re, hex_src_psi_im, t, y, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_fHex,Nsnk_f,Nperms_f);
            make_hex_dibaryon_correlator(H_BB_r2_re, H_BB_r2_im, snk_B1_Blocal_r1_re, snk_B1_Blocal_r1_im, snk_B2_Blocal_r2_re, snk_B2_Blocal_r2_im, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_r2_1, snk_spin_weights_r2_1, snk_weights_r2_1, hex_src_psi_re, hex_src_psi_im, t, y, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_fHex,Nsnk_f,Nperms_f);
            make_hex_dibaryon_correlator(H_BB_r2_re, H_BB_r2_im, snk_B1_Blocal_r1_re, snk_B1_Blocal_r1_im, snk_B2_Blocal_r2_re, snk_B2_Blocal_r2_im, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_r2_2, snk_spin_weights_r2_2, snk_weights_r2_2, hex_src_psi_re, hex_src_psi_im, t, y, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_fHex,Nsnk_f,Nperms_f);
            make_hex_dibaryon_correlator(H_BB_r2_re, H_BB_r2_im, snk_B1_Blocal_r2_re, snk_B1_Blocal_r2_im, snk_B2_Blocal_r1_re, snk_B2_Blocal_r1_im, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_r2_1, snk_spin_weights_r2_1, snk_weights_r2_1, hex_src_psi_re, hex_src_psi_im, t, y, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_fHex,Nsnk_f,Nperms_f);
            make_hex_dibaryon_correlator(H_BB_r2_re, H_BB_r2_im, snk_B1_Blocal_r2_re, snk_B1_Blocal_r2_im, snk_B2_Blocal_r1_re, snk_B2_Blocal_r1_im, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_r2_2, snk_spin_weights_r2_2, snk_weights_r2_2, hex_src_psi_re, hex_src_psi_im, t, y, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_fHex,Nsnk_f,Nperms_f);
            make_hex_dibaryon_correlator(H_BB_r3_re, H_BB_r3_im, snk_B1_Blocal_r2_re, snk_B1_Blocal_r2_im, snk_B2_Blocal_r2_re, snk_B2_Blocal_r2_im, perms, sigs, overall_weight, snk_color_weights_r3, snk_spin_weights_r3, snk_weights_r3, hex_src_psi_re, hex_src_psi_im, t, y, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_fHex,Nsnk_f,Nperms_f);
            if (space_symmetric != 0) {
               make_hex_dibaryon_correlator(H_BB_0_re, H_BB_0_im, snk_B2_Blocal_r1_re, snk_B2_Blocal_r1_im, snk_B1_Blocal_r2_re, snk_B1_Blocal_r2_im, perms, sigs, space_symmetric*overall_weight/sqrt(2.0), snk_color_weights_1, snk_spin_weights_1, snk_weights_1, hex_src_psi_re, hex_src_psi_im, t, y, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_fHex,Nsnk_f,Nperms_f);
               make_hex_dibaryon_correlator(H_BB_0_re, H_BB_0_im, snk_B2_Blocal_r1_re, snk_B2_Blocal_r1_im, snk_B1_Blocal_r2_re, snk_B1_Blocal_r2_im, perms, sigs, space_symmetric*overall_weight/sqrt(2.0), snk_color_weights_2, snk_spin_weights_2, snk_weights_2, hex_src_psi_re, hex_src_psi_im, t, y, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_fHex,Nsnk_f,Nperms_f);
               make_hex_dibaryon_correlator(H_BB_0_re, H_BB_0_im, snk_B2_Blocal_r2_re, snk_B2_Blocal_r2_im, snk_B1_Blocal_r1_re, snk_B1_Blocal_r1_im, perms, sigs, -1.0*space_symmetric*overall_weight/sqrt(2.0), snk_color_weights_1, snk_spin_weights_1, snk_weights_1, hex_src_psi_re, hex_src_psi_im, t, y, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_fHex,Nsnk_f,Nperms_f);
               make_hex_dibaryon_correlator(H_BB_0_re, H_BB_0_im, snk_B2_Blocal_r2_re, snk_B2_Blocal_r2_im, snk_B1_Blocal_r1_re, snk_B1_Blocal_r1_im, perms, sigs, -1.0*space_symmetric*overall_weight/sqrt(2.0), snk_color_weights_2, snk_spin_weights_2, snk_weights_2, hex_src_psi_re, hex_src_psi_im, t, y, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_fHex,Nsnk_f,Nperms_f);
               make_hex_dibaryon_correlator(H_BB_r1_re, H_BB_r1_im, snk_B2_Blocal_r1_re, snk_B2_Blocal_r1_im, snk_B1_Blocal_r1_re, snk_B1_Blocal_r1_im, perms, sigs, space_symmetric*overall_weight, snk_color_weights_r1, snk_spin_weights_r1, snk_weights_r1, hex_src_psi_re, hex_src_psi_im, t, y, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_fHex,Nsnk_f,Nperms_f);
               make_hex_dibaryon_correlator(H_BB_r2_re, H_BB_r2_im, snk_B2_Blocal_r1_re, snk_B2_Blocal_r1_im, snk_B1_Blocal_r2_re, snk_B1_Blocal_r2_im, perms, sigs, space_symmetric*overall_weight/sqrt(2.0), snk_color_weights_r2_1, snk_spin_weights_r2_1, snk_weights_r2_1, hex_src_psi_re, hex_src_psi_im, t, y, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_fHex,Nsnk_f,Nperms_f);
               make_hex_dibaryon_correlator(H_BB_r2_re, H_BB_r2_im, snk_B2_Blocal_r1_re, snk_B2_Blocal_r1_im, snk_B1_Blocal_r2_re, snk_B1_Blocal_r2_im, perms, sigs, space_symmetric*overall_weight/sqrt(2.0), snk_color_weights_r2_2, snk_spin_weights_r2_2, snk_weights_r2_2, hex_src_psi_re, hex_src_psi_im, t, y, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_fHex,Nsnk_f,Nperms_f);
               make_hex_dibaryon_correlator(H_BB_r2_re, H_BB_r2_im, snk_B2_Blocal_r2_re, snk_B2_Blocal_r2_im, snk_B1_Blocal_r1_re, snk_B1_Blocal_r1_im, perms, sigs, space_symmetric*overall_weight/sqrt(2.0), snk_color_weights_r2_1, snk_spin_weights_r2_1, snk_weights_r2_1, hex_src_psi_re, hex_src_psi_im, t, y, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_fHex,Nsnk_f,Nperms_f);
               make_hex_dibaryon_correlator(H_BB_r2_re, H_BB_r2_im, snk_B2_Blocal_r2_re, snk_B2_Blocal_r2_im, snk_B1_Blocal_r1_re, snk_B1_Blocal_r1_im, perms, sigs, space_symmetric*overall_weight/sqrt(2.0), snk_color_weights_r2_2, snk_spin_weights_r2_2, snk_weights_r2_2, hex_src_psi_re, hex_src_psi_im, t, y, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_fHex,Nsnk_f,Nperms_f);
               make_hex_dibaryon_correlator(H_BB_r3_re, H_BB_r3_im, snk_B2_Blocal_r2_re, snk_B2_Blocal_r2_im, snk_B1_Blocal_r2_re, snk_B1_Blocal_r2_im, perms, sigs, space_symmetric*overall_weight, snk_color_weights_r3, snk_spin_weights_r3, snk_weights_r3, hex_src_psi_re, hex_src_psi_im, t, y, Nc_f,Ns_f,Vsrc_f,Vsnk_f,Nt_f,Nw_f,Nq_f,Nsrc_fHex,Nsnk_f,Nperms_f);
            }
         }
         for (m=0; m<Nsrc_fHex; m++) {
            for (n=0; n<Nsnk_f; n++) {
               C_re[index_4d(0,Nsrc_f+m,n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] = H_BB_0_re[index_3d(m,n,t,Nsnk_f,Nt_f)];
               C_im[index_4d(0,Nsrc_f+m,n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] = H_BB_0_im[index_3d(m,n,t,Nsnk_f,Nt_f)];
               C_re[index_4d(1,Nsrc_f+m,n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] = H_BB_r1_re[index_3d(m,n,t,Nsnk_f,Nt_f)];
               C_im[index_4d(1,Nsrc_f+m,n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] = H_BB_r1_im[index_3d(m,n,t,Nsnk_f,Nt_f)];
               C_re[index_4d(2,Nsrc_f+m,n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] = H_BB_r2_re[index_3d(m,n,t,Nsnk_f,Nt_f)];
               C_im[index_4d(2,Nsrc_f+m,n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] = H_BB_r2_im[index_3d(m,n,t,Nsnk_f,Nt_f)];
               C_re[index_4d(3,Nsrc_f+m,n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] = H_BB_r3_re[index_3d(m,n,t,Nsnk_f,Nt_f)];
               C_im[index_4d(3,Nsrc_f+m,n,t ,Nsrc_f+Nsrc_fHex,Nsnk_f+Nsnk_fHex,Nt_f)] = H_BB_r3_im[index_3d(m,n,t,Nsnk_f,Nt_f)];
            }
         }
      }
      free(snk_B1_Blocal_r1_re);
      free(snk_B1_Blocal_r1_im);
      free(snk_B1_Blocal_r2_re);
      free(snk_B1_Blocal_r2_im);
      free(snk_B2_Blocal_r1_re);
      free(snk_B2_Blocal_r1_im);
      free(snk_B2_Blocal_r2_re);
      free(snk_B2_Blocal_r2_im);
      printf("made H-BB \n");
   }
   free(H_BB_0_re);
   free(H_BB_0_im);
   free(H_BB_r1_re);
   free(H_BB_r1_im);
   free(H_BB_r2_re);
   free(H_BB_r2_im);
   free(H_BB_r3_re);
   free(H_BB_r3_im);
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
               for (n=0; n<Nsnk_f; n++) {
                  std::complex<double> psi(snk_psi_re[index_2d(x,n ,Nsnk_f)], snk_psi_im[index_2d(x,n ,Nsnk_f)]);
                  std::complex<double> corr = block * psi;
                  C_re[index_4d(0,m,n,t ,Nsrc_f,Nsnk_f,Nt_f)] += src_weights_r1[wnum] * real(corr);
                  C_im[index_4d(0,m,n,t ,Nsrc_f,Nsnk_f,Nt_f)] += src_weights_r1[wnum] * imag(corr);
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
               std::complex<double> block(Blocal_r2_re[Blocal_index(iC,iS,kC,kS,jC,jS,m ,Nc_f,Ns_f,Nsrc_f)], Blocal_r2_im[Blocal_index(iC,iS,kC,kS,jC,jS,m ,Nc_f,Ns_f,Nsrc_f)]);
               for (n=0; n<Nsnk_f; n++) {
                  std::complex<double> psi(snk_psi_re[index_2d(x,n ,Nsnk_f)], snk_psi_im[index_2d(x,n ,Nsnk_f)]);
                  std::complex<double> corr = block * psi;
                  C_re[index_4d(1,m,n,t ,Nsrc_f,Nsnk_f,Nt_f)] += src_weights_r2[wnum] * real(corr);
                  C_im[index_4d(1,m,n,t ,Nsrc_f,Nsnk_f,Nt_f)] += src_weights_r2[wnum] * imag(corr);
               }
            }
         }
      }
   } 
   /* clean up */
   free(Blocal_r1_re);
   free(Blocal_r1_im);
   free(Blocal_r2_re);
   free(Blocal_r2_im); 
}
