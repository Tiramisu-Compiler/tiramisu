#include <stdio.h> 
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

#include "qblocks_2pt_scalar.h"                                       /* DEPS */
#include "tiramisu_wrapper.h"

/* index functions */
int index_2d(int a, int b, int length2) {
   return b +length2*( a );
}
int index_3d(int a, int b, int c, int length2, int length3) {
   return c +length3*( b +length2*( a ));
}
int index_4d(int a, int b, int c, int d, int length2, int length3, int length4) {
   return d +length4*( c +length3*( b +length2*( a )));
}
int prop_index(int q, int t, int c1, int s1, int c2, int s2, int y, int x, int Nc, int Ns, int Vsrc, int Vsnk, int Nt) {
   return x +Vsnk*( y +Vsrc*( c2 +Nc*( s2 +Ns*( c1 +Nc*( s1 +Ns*( t +Nt* q ))))));
}
int Q_index(int t, int c1, int s1, int c2, int s2, int x1, int c3, int s3, int y, int Nc, int Ns, int Vsrc, int Vsnk) {
   return y +Vsrc*( s3 +Ns*( c3 +Nc*( x1 +Vsnk*( s2 +Ns*( c2 +Nc*( s1 +Ns*( c1 +Nc*( t ))))))));
}
int Blocal_index(int t, int c1, int s1, int c2, int s2, int x, int c3, int s3, int m, int Nc, int Ns, int Vsnk, int Nsrc) {
   return m +Nsrc*( s3 +Ns*( c3 +Nc*( x +Vsnk*( s2 +Ns*( c2 +Nc*( s1 +Ns*( c1 +Nc*( t ))))))));
}
int tBlocal_index(int t, int c1, int s1, int c2, int s2, int x, int m, int c3, int s3, int Nc, int Ns, int Vsnk, int Nsrc) {
   return s3 +Ns*( c3 + Nc*( m +Nsrc*( x +Vsnk*( s2 +Ns*( c2 +Nc*( s1 +Ns*( c1 +Nc*( t ))))))));
}

int Bdouble_index(int t, int c1, int s1, int c2, int s2, int x1, int c3, int s3, int x2, int m, int Nc, int Ns, int Vsnk, int Nsrc) {
   return m +Nsrc*( x2 +Vsnk*(  s3 +Ns*( c3 +Nc*( x1 +Vsnk*( s2 +Ns*( c2 +Nc*( s1 +Ns*( c1 +Nc*( t )))))))));
}

void error_msg(char *msg)
{
    printf("\nError! %s.\n", msg);
    exit(1);
}

double rtclock()
{
    struct timeval Tp;
    gettimeofday(&Tp, NULL);

    return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
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
    const int Nc,
    const int Ns,
    const int Vsrc,
    const int Vsnk,
    const int Nt,
    const int Nw,
    const int Nq,
    const int Nsrc) {
   /* loop indices */
   int iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, iC, iS, jC, jS, kC, kS, x, t, y, wnum, m;
   /* subexpressions */
   double prop_prod_02_re;
   double prop_prod_02_im;
   double prop_prod_re;
   double prop_prod_im;
   /* initialize */
   for (t=0; t<Nt; t++) {
      for (iCprime=0; iCprime<Nc; iCprime++) {
         for (iSprime=0; iSprime<Ns; iSprime++) {
            for (kCprime=0; kCprime<Nc; kCprime++) {
               for (kSprime=0; kSprime<Ns; kSprime++) {
                  for (x=0; x<Vsnk; x++) {
                     for (jCprime=0; jCprime<Nc; jCprime++) {
                        for (jSprime=0; jSprime<Ns; jSprime++) {
                           for (m=0; m<Nsrc; m++) {
                              Blocal_re[Blocal_index(t,iCprime,iSprime,kCprime,kSprime,x,jCprime,jSprime,m ,Nc,Ns,Vsnk,Nsrc)] = 0.0;
                              Blocal_im[Blocal_index(t,iCprime,iSprime,kCprime,kSprime,x,jCprime,jSprime,m ,Nc,Ns,Vsnk,Nsrc)] = 0.0;
                           }
                        }
                     }
                  }
               }
            }
         }
      }
   }
   /* build local (no quark exchange) block */
   for (wnum=0; wnum<Nw; wnum++) {
      iC = color_weights[index_2d(wnum,0, Nq)];
      iS = spin_weights[index_2d(wnum,0, Nq)];
      jC = color_weights[index_2d(wnum,1, Nq)];
      jS = spin_weights[index_2d(wnum,1, Nq)];
      kC = color_weights[index_2d(wnum,2, Nq)];
      kS = spin_weights[index_2d(wnum,2, Nq)];
      for (t=0; t<Nt; t++) {
         for (iCprime=0; iCprime<Nc; iCprime++) {
            for (iSprime=0; iSprime<Ns; iSprime++) {
               for (kCprime=0; kCprime<Nc; kCprime++) {
                  for (kSprime=0; kSprime<Ns; kSprime++) {
                     for (x=0; x<Vsnk; x++) {
                        for (y=0; y<Vsrc; y++) {
                           prop_prod_02_re = weights[wnum] * ( (prop_re[prop_index(0,t,iC,iS,iCprime,iSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] * prop_re[prop_index(2,t,kC,kS,kCprime,kSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] - prop_im[prop_index(0,t,iC,iS,iCprime,iSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] * prop_im[prop_index(2,t,kC,kS,kCprime,kSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)]) - (prop_re[prop_index(0,t,iC,iS,kCprime,kSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] * prop_re[prop_index(2,t,kC,kS,iCprime,iSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] - prop_im[prop_index(0,t,iC,iS,kCprime,kSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] * prop_im[prop_index(2,t,kC,kS,iCprime,iSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)]) );
                           prop_prod_02_im = weights[wnum] * ( (prop_re[prop_index(0,t,iC,iS,iCprime,iSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] * prop_im[prop_index(2,t,kC,kS,kCprime,kSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] + prop_im[prop_index(0,t,iC,iS,iCprime,iSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] * prop_re[prop_index(2,t,kC,kS,kCprime,kSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)]) - (prop_re[prop_index(0,t,iC,iS,kCprime,kSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] * prop_im[prop_index(2,t,kC,kS,iCprime,iSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] + prop_im[prop_index(0,t,iC,iS,kCprime,kSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] * prop_re[prop_index(2,t,kC,kS,iCprime,iSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)]) );
                           for (jCprime=0; jCprime<Nc; jCprime++) {
                              for (jSprime=0; jSprime<Ns; jSprime++) {
                                 prop_prod_re = prop_prod_02_re * prop_re[prop_index(1,t,jC,jS,jCprime,jSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] - prop_prod_02_im * prop_im[prop_index(1,t,jC,jS,jCprime,jSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)];
                                 prop_prod_im = prop_prod_02_re * prop_im[prop_index(1,t,jC,jS,jCprime,jSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] + prop_prod_02_im * prop_re[prop_index(1,t,jC,jS,jCprime,jSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)];
                                 for (m=0; m<Nsrc; m++) {
                                    Blocal_re[Blocal_index(t,iCprime,iSprime,kCprime,kSprime,x,jCprime,jSprime,m ,Nc,Ns,Vsnk,Nsrc)] += psi_re[index_2d(y,m ,Nsrc)] * prop_prod_re - psi_im[index_2d(y,m ,Nsrc)] * prop_prod_im;
                                    Blocal_im[Blocal_index(t,iCprime,iSprime,kCprime,kSprime,x,jCprime,jSprime,m ,Nc,Ns,Vsnk,Nsrc)] += psi_re[index_2d(y,m ,Nsrc)] * prop_prod_im + psi_im[index_2d(y,m ,Nsrc)] * prop_prod_re;
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
    const int Nc,
    const int Ns,
    const int Vsrc,
    const int Vsnk,
    const int Nt,
    const int Nw,
    const int Nq,
    const int Nsnk) {
   /* loop indices */
   int iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, iC, iS, jC, jS, kC, kS, x, t, y, wnum, n;
   /* subexpressions */
   double prop_prod_02_re;
   double prop_prod_02_im;
   double prop_prod_re;
   double prop_prod_im;
   /* initialize */
   for (t=0; t<Nt; t++) {
      for (iCprime=0; iCprime<Nc; iCprime++) {
         for (iSprime=0; iSprime<Ns; iSprime++) {
            for (kCprime=0; kCprime<Nc; kCprime++) {
               for (kSprime=0; kSprime<Ns; kSprime++) {
                  for (y=0; y<Vsrc; y++) {
                     for (jCprime=0; jCprime<Nc; jCprime++) {
                        for (jSprime=0; jSprime<Ns; jSprime++) {
                           for (n=0; n<Nsnk; n++) {
                              Blocal_re[Blocal_index(t,iCprime,iSprime,kCprime,kSprime,y,jCprime,jSprime,n ,Nc,Ns,Vsrc,Nsnk)] = 0.0;
                              Blocal_im[Blocal_index(t,iCprime,iSprime,kCprime,kSprime,y,jCprime,jSprime,n ,Nc,Ns,Vsrc,Nsnk)] = 0.0;
                           }
                        }
                     }
                  }
               }
            }
         }
      }
   }
   /* build local (no quark exchange) block */
   for (wnum=0; wnum<Nw; wnum++) {
      iC = color_weights[index_2d(wnum,0, Nq)];
      iS = spin_weights[index_2d(wnum,0, Nq)];
      jC = color_weights[index_2d(wnum,1, Nq)];
      jS = spin_weights[index_2d(wnum,1, Nq)];
      kC = color_weights[index_2d(wnum,2, Nq)];
      kS = spin_weights[index_2d(wnum,2, Nq)];
      for (t=0; t<Nt; t++) {
         for (iCprime=0; iCprime<Nc; iCprime++) {
            for (iSprime=0; iSprime<Ns; iSprime++) {
               for (kCprime=0; kCprime<Nc; kCprime++) {
                  for (kSprime=0; kSprime<Ns; kSprime++) {
                     for (y=0; y<Vsrc; y++) {
                        for (x=0; x<Vsnk; x++) {
                           prop_prod_02_re = weights[wnum] * ( (prop_re[prop_index(0,t,iC,iS,iCprime,iSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] * prop_re[prop_index(2,t,kC,kS,kCprime,kSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] - prop_im[prop_index(0,t,iC,iS,iCprime,iSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] * prop_im[prop_index(2,t,kC,kS,kCprime,kSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)]) - (prop_re[prop_index(0,t,iC,iS,kCprime,kSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] * prop_re[prop_index(2,t,kC,kS,iCprime,iSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] - prop_im[prop_index(0,t,iC,iS,kCprime,kSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] * prop_im[prop_index(2,t,kC,kS,iCprime,iSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)]) );
                           prop_prod_02_im = weights[wnum] * ( (prop_re[prop_index(0,t,iC,iS,iCprime,iSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] * prop_im[prop_index(2,t,kC,kS,kCprime,kSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] + prop_im[prop_index(0,t,iC,iS,iCprime,iSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] * prop_re[prop_index(2,t,kC,kS,kCprime,kSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)]) - (prop_re[prop_index(0,t,iC,iS,kCprime,kSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] * prop_im[prop_index(2,t,kC,kS,iCprime,iSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] + prop_im[prop_index(0,t,iC,iS,kCprime,kSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] * prop_re[prop_index(2,t,kC,kS,iCprime,iSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)]) );
                           for (jCprime=0; jCprime<Nc; jCprime++) {
                              for (jSprime=0; jSprime<Ns; jSprime++) {
                                 prop_prod_re = prop_prod_02_re * prop_re[prop_index(1,t,jC,jS,jCprime,jSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] - prop_prod_02_im * prop_im[prop_index(1,t,jC,jS,jCprime,jSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)];
                                 prop_prod_im = prop_prod_02_re * prop_im[prop_index(1,t,jC,jS,jCprime,jSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] + prop_prod_02_im * prop_re[prop_index(1,t,jC,jS,jCprime,jSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)];
                                 for (n=0; n<Nsnk; n++) {
                                    Blocal_re[Blocal_index(t,iCprime,iSprime,kCprime,kSprime,y,jCprime,jSprime,n ,Nc,Ns,Vsrc,Nsnk)] += psi_re[index_2d(x,n ,Nsnk)] * prop_prod_re - psi_im[index_2d(x,n ,Nsnk)] * prop_prod_im;
                                    Blocal_im[Blocal_index(t,iCprime,iSprime,kCprime,kSprime,y,jCprime,jSprime,n ,Nc,Ns,Vsrc,Nsnk)] += psi_re[index_2d(x,n ,Nsnk)] * prop_prod_im + psi_im[index_2d(x,n ,Nsnk)] * prop_prod_re;
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
   }
}

void make_single_block(double* Bsingle_re, 
    double* Bsingle_im, 
    const double* prop_re,
    const double* prop_im, 
    const int* color_weights, 
    const int* spin_weights, 
    const double* weights, 
    const double* psi_re,
    const double* psi_im,
    const int Nc,
    const int Ns,
    const int Vsrc,
    const int Vsnk,
    const int Nt,
    const int Nw,
    const int Nq,
    const int Nsrc) {
   /* indices */
   int iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, iC, iS, jC, jS, kC, kS, x, x1, x2, t, y, wnum, m;
   /* subexpressions */
   double prop_prod_re;
   double prop_prod_im;
   double* Q02_re = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Vsrc * sizeof (double));
   double* Q02_im = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Vsrc * sizeof (double));
   /* initialize */
   for (t=0; t<Nt; t++) {
      for (iCprime=0; iCprime<Nc; iCprime++) {
         for (iSprime=0; iSprime<Ns; iSprime++) {
            for (kCprime=0; kCprime<Nc; kCprime++) {
               for (kSprime=0; kSprime<Ns; kSprime++) {
                  for (x1=0; x1<Vsnk; x1++) {
                     for (m=0; m<Nsrc; m++) {
                        for (jCprime=0; jCprime<Nc; jCprime++) {
                           for (jSprime=0; jSprime<Ns; jSprime++) {
                              for (x2=0; x2<Vsnk; x2++) {
                                 Bsingle_re[Bdouble_index(t,iCprime,iSprime,kCprime,kSprime,x1,jCprime,jSprime,x2,m ,Nc,Ns,Vsnk,Nsrc)] = 0.0;
                                 Bsingle_im[Bdouble_index(t,iCprime,iSprime,kCprime,kSprime,x1,jCprime,jSprime,x2,m ,Nc,Ns,Vsnk,Nsrc)] = 0.0;
                              }
                           }
                        }
                     }
                     for (jC=0; jC<Nc; jC++) {
                        for (jS=0; jS<Ns; jS++) {
                           for (y=0; y<Vsrc; y++) {
                              Q02_re[Q_index(t,iCprime,iSprime,kCprime,kSprime,x1,jC,jS,y ,Nc,Ns,Vsrc,Vsnk)] = 0.0;
                              Q02_im[Q_index(t,iCprime,iSprime,kCprime,kSprime,x1,jC,jS,y ,Nc,Ns,Vsrc,Vsnk)] = 0.0;
                           }
                        }
                     }
                  }
               }
            }
         }
      }
   }
   /* build diquarks */
   for (wnum=0; wnum<Nw; wnum++) {
      iC = color_weights[index_2d(wnum,0, Nq)];
      iS = spin_weights[index_2d(wnum,0, Nq)];
      jC = color_weights[index_2d(wnum,1, Nq)];
      jS = spin_weights[index_2d(wnum,1, Nq)];
      kC = color_weights[index_2d(wnum,2, Nq)];
      kS = spin_weights[index_2d(wnum,2, Nq)];
      for (t=0; t<Nt; t++) {
         for (iCprime=0; iCprime<Nc; iCprime++) {
            for (iSprime=0; iSprime<Ns; iSprime++) {
               for (kCprime=0; kCprime<Nc; kCprime++) {
                  for (kSprime=0; kSprime<Ns; kSprime++) {
                     for (y=0; y<Vsrc; y++) {
                        for (x=0; x<Vsnk; x++) {
                           Q02_re[Q_index(t,iCprime,iSprime,kCprime,kSprime,x,jC,jS,y ,Nc,Ns,Vsrc,Vsnk)] += weights[wnum] * ( (prop_re[prop_index(0,t,iC,iS,iCprime,iSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] * prop_re[prop_index(2,t,kC,kS,kCprime,kSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] - prop_im[prop_index(0,t,iC,iS,iCprime,iSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] * prop_im[prop_index(2,t,kC,kS,kCprime,kSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)]) - (prop_re[prop_index(0,t,iC,iS,kCprime,kSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] * prop_re[prop_index(2,t,kC,kS,iCprime,iSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] - prop_im[prop_index(0,t,iC,iS,kCprime,kSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] * prop_im[prop_index(2,t,kC,kS,iCprime,iSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)]) );
                           Q02_im[Q_index(t,iCprime,iSprime,kCprime,kSprime,x,jC,jS,y ,Nc,Ns,Vsrc,Vsnk)] += weights[wnum] * ( (prop_re[prop_index(0,t,iC,iS,iCprime,iSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] * prop_im[prop_index(2,t,kC,kS,kCprime,kSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] + prop_im[prop_index(0,t,iC,iS,iCprime,iSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] * prop_re[prop_index(2,t,kC,kS,kCprime,kSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)]) - (prop_re[prop_index(0,t,iC,iS,kCprime,kSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] * prop_im[prop_index(2,t,kC,kS,iCprime,iSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] + prop_im[prop_index(0,t,iC,iS,kCprime,kSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] * prop_re[prop_index(2,t,kC,kS,iCprime,iSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)]) );
                        }
                     }
                  }
               }
            }
         }
      }
   }
   /* build q2-exchange block */
   for (t=0; t<Nt; t++) {
      for (iCprime=0; iCprime<Nc; iCprime++) {
         for (iSprime=0; iSprime<Ns; iSprime++) {
            for (kCprime=0; kCprime<Nc; kCprime++) {
               for (kSprime=0; kSprime<Ns; kSprime++) {
                  for (x1=0; x1<Vsnk; x1++) {
                     for (jC=0; jC<Nc; jC++) {
                        for (jS=0; jS<Ns; jS++) {
                           for (jCprime=0; jCprime<Nc; jCprime++) {
                              for (jSprime=0; jSprime<Ns; jSprime++) {
                                 for (y=0; y<Vsrc; y++) {
                                    for (x2=0; x2<Vsnk; x2++) {
                                       prop_prod_re = Q02_re[Q_index(t,iCprime,iSprime,kCprime,kSprime,x1,jC,jS,y ,Nc,Ns,Vsrc,Vsnk)] * prop_re[prop_index(1,t,jC,jS,jCprime,jSprime,y,x2 ,Nc,Ns,Vsrc,Vsnk,Nt)] - Q02_im[Q_index(t,iCprime,iSprime,kCprime,kSprime,x1,jC,jS,y ,Nc,Ns,Vsrc,Vsnk)] * prop_im[prop_index(1,t,jC,jS,jCprime,jSprime,y,x2 ,Nc,Ns,Vsrc,Vsnk,Nt)];
                                       prop_prod_im = Q02_re[Q_index(t,iCprime,iSprime,kCprime,kSprime,x1,jC,jS,y ,Nc,Ns,Vsrc,Vsnk)] * prop_im[prop_index(1,t,jC,jS,jCprime,jSprime,y,x2 ,Nc,Ns,Vsrc,Vsnk,Nt)] + Q02_im[Q_index(t,iCprime,iSprime,kCprime,kSprime,x1,jC,jS,y ,Nc,Ns,Vsrc,Vsnk)] * prop_re[prop_index(1,t,jC,jS,jCprime,jSprime,y,x2 ,Nc,Ns,Vsrc,Vsnk,Nt)];
                                       for (m=0; m<Nsrc; m++) {
                                          Bsingle_re[Bdouble_index(t,iCprime,iSprime,kCprime,kSprime,x1,jCprime,jSprime,x2,m ,Nc,Ns,Vsnk,Nsrc)] += psi_re[index_2d(y,m ,Nsrc)] * prop_prod_re - psi_im[index_2d(y,m ,Nsrc)] * prop_prod_im;
                                          Bsingle_im[Bdouble_index(t,iCprime,iSprime,kCprime,kSprime,x1,jCprime,jSprime,x2,m ,Nc,Ns,Vsnk,Nsrc)] += psi_re[index_2d(y,m ,Nsrc)] * prop_prod_im + psi_im[index_2d(y,m ,Nsrc)] * prop_prod_re;
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
         }
      }
   }
   free(Q02_re);
   free(Q02_im);
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
    const int Nc,
    const int Ns,
    const int Vsrc,
    const int Vsnk,
    const int Nt,
    const int Nw,
    const int Nq,
    const int Nsrc) {
   /* indices */
   int iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, iC, iS, jC, jS, kC, kS, x, x1, x2, t, y, wnum, n;
   /* subexpressions */
   double prop_prod_re;
   double prop_prod_im;
   double* Q12_re = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Vsrc * sizeof (double));
   double* Q12_im = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Vsrc * sizeof (double));
   double* Q01_re = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Vsrc * sizeof (double));
   double* Q01_im = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Vsrc * sizeof (double));
   /* initialize */
   for (t=0; t<Nt; t++) {
      for (iCprime=0; iCprime<Nc; iCprime++) {
         for (iSprime=0; iSprime<Ns; iSprime++) {
            for (kCprime=0; kCprime<Nc; kCprime++) {
               for (kSprime=0; kSprime<Ns; kSprime++) {
                  for (x1=0; x1<Vsnk; x1++) {
                     for (n=0; n<Nsrc; n++) {
                        for (jCprime=0; jCprime<Nc; jCprime++) {
                           for (jSprime=0; jSprime<Ns; jSprime++) {
                              for (x2=0; x2<Vsnk; x2++) {
                                 Bdouble_re[Bdouble_index(t,iCprime,iSprime,kCprime,kSprime,x1,jCprime,jSprime,x2,n ,Nc,Ns,Vsnk,Nsrc)] = 0.0;
                                 Bdouble_im[Bdouble_index(t,iCprime,iSprime,kCprime,kSprime,x1,jCprime,jSprime,x2,n ,Nc,Ns,Vsnk,Nsrc)] = 0.0;
                              }
                           }
                        }
                     }
                     for (jC=0; jC<Nc; jC++) {
                        for (jS=0; jS<Ns; jS++) {
                           for (y=0; y<Vsrc; y++) {
                              Q12_re[Q_index(t,iCprime,iSprime,kCprime,kSprime,x1,jC,jS,y ,Nc,Ns,Vsrc,Vsnk)] = 0.0;
                              Q12_im[Q_index(t,iCprime,iSprime,kCprime,kSprime,x1,jC,jS,y ,Nc,Ns,Vsrc,Vsnk)] = 0.0;
                              Q01_re[Q_index(t,iCprime,iSprime,kCprime,kSprime,x1,jC,jS,y ,Nc,Ns,Vsrc,Vsnk)] = 0.0;
                              Q01_im[Q_index(t,iCprime,iSprime,kCprime,kSprime,x1,jC,jS,y ,Nc,Ns,Vsrc,Vsnk)] = 0.0;
                           }
                        }
                     }
                  }
               }
            }
         }
      }
   }
   /* build diquarks */
   for (wnum=0; wnum<Nw; wnum++) {
      iC = color_weights[index_2d(wnum,0, Nq)];
      iS = spin_weights[index_2d(wnum,0, Nq)];
      jC = color_weights[index_2d(wnum,1, Nq)];
      jS = spin_weights[index_2d(wnum,1, Nq)];
      kC = color_weights[index_2d(wnum,2, Nq)];
      kS = spin_weights[index_2d(wnum,2, Nq)];
      for (t=0; t<Nt; t++) {
         for (jCprime=0; jCprime<Nc; jCprime++) {
            for (jSprime=0; jSprime<Ns; jSprime++) {
               for (kCprime=0; kCprime<Nc; kCprime++) {
                  for (kSprime=0; kSprime<Ns; kSprime++) {
                     for (y=0; y<Vsrc; y++) {
                        for (x=0; x<Vsnk; x++) {
                           Q12_re[Q_index(t,jCprime,jSprime,kCprime,kSprime,x,iC,iS,y ,Nc,Ns,Vsrc,Vsnk)] += weights[wnum] * (prop_re[prop_index(1,t,jC,jS,jCprime,jSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] * prop_re[prop_index(2,t,kC,kS,kCprime,kSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] - prop_im[prop_index(1,t,jC,jS,jCprime,jSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] * prop_im[prop_index(2,t,kC,kS,kCprime,kSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)]);
                           Q12_im[Q_index(t,jCprime,jSprime,kCprime,kSprime,x,iC,iS,y ,Nc,Ns,Vsrc,Vsnk)] += weights[wnum] * (prop_re[prop_index(1,t,jC,jS,jCprime,jSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] * prop_im[prop_index(2,t,kC,kS,kCprime,kSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] + prop_im[prop_index(1,t,jC,jS,jCprime,jSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] * prop_re[prop_index(2,t,kC,kS,kCprime,kSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)]);
                           Q01_re[Q_index(t,jCprime,jSprime,kCprime,kSprime,x,kC,kS,y ,Nc,Ns,Vsrc,Vsnk)] += weights[wnum] * (prop_re[prop_index(0,t,iC,iS,kCprime,kSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] * prop_re[prop_index(1,t,jC,jS,jCprime,jSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] - prop_im[prop_index(0,t,iC,iS,kCprime,kSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] * prop_im[prop_index(1,t,jC,jS,jCprime,jSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)]);
                           Q01_im[Q_index(t,jCprime,jSprime,kCprime,kSprime,x,kC,kS,y ,Nc,Ns,Vsrc,Vsnk)] += weights[wnum] * (prop_re[prop_index(0,t,iC,iS,kCprime,kSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] * prop_im[prop_index(1,t,jC,jS,jCprime,jSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] + prop_im[prop_index(0,t,iC,iS,kCprime,kSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] * prop_re[prop_index(1,t,jC,jS,jCprime,jSprime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)]);
                        }
                     }
                  }
               }
            }
         }
      }
   }
   /* build q2-exchange block */
   for (t=0; t<Nt; t++) {
      for (jCprime=0; jCprime<Nc; jCprime++) {
         for (jSprime=0; jSprime<Ns; jSprime++) {
            for (kCprime=0; kCprime<Nc; kCprime++) {
               for (kSprime=0; kSprime<Ns; kSprime++) {
                  for (x1=0; x1<Vsnk; x1++) {
                     for (iC=0; iC<Nc; iC++) {
                        for (iS=0; iS<Ns; iS++) {
                           for (iCprime=0; iCprime<Nc; iCprime++) {
                              for (iSprime=0; iSprime<Ns; iSprime++) {
                                 for (y=0; y<Vsrc; y++) {
                                    for (x2=0; x2<Vsnk; x2++) {
                                       prop_prod_re = Q12_re[Q_index(t,jCprime,jSprime,kCprime,kSprime,x1,iC,iS,y ,Nc,Ns,Vsrc,Vsnk)] * prop_re[prop_index(0,t,iC,iS,iCprime,iSprime,y,x2 ,Nc,Ns,Vsrc,Vsnk,Nt)] - Q12_im[Q_index(t,jCprime,jSprime,kCprime,kSprime,x1,iC,iS,y ,Nc,Ns,Vsrc,Vsnk)] * prop_im[prop_index(0,t,iC,iS,iCprime,iSprime,y,x2 ,Nc,Ns,Vsrc,Vsnk,Nt)];
                                       prop_prod_im = Q12_re[Q_index(t,jCprime,jSprime,kCprime,kSprime,x1,iC,iS,y ,Nc,Ns,Vsrc,Vsnk)] * prop_im[prop_index(0,t,iC,iS,iCprime,iSprime,y,x2 ,Nc,Ns,Vsrc,Vsnk,Nt)] + Q12_im[Q_index(t,jCprime,jSprime,kCprime,kSprime,x1,iC,iS,y ,Nc,Ns,Vsrc,Vsnk)] * prop_re[prop_index(0,t,iC,iS,iCprime,iSprime,y,x2 ,Nc,Ns,Vsrc,Vsnk,Nt)];
                                       prop_prod_re -= Q01_re[Q_index(t,jCprime,jSprime,kCprime,kSprime,x1,iC,iS,y ,Nc,Ns,Vsrc,Vsnk)] * prop_re[prop_index(2,t,iC,iS,iCprime,iSprime,y,x2 ,Nc,Ns,Vsrc,Vsnk,Nt)] - Q01_im[Q_index(t,jCprime,jSprime,kCprime,kSprime,x1,iC,iS,y ,Nc,Ns,Vsrc,Vsnk)] * prop_im[prop_index(2,t,iC,iS,iCprime,iSprime,y,x2 ,Nc,Ns,Vsrc,Vsnk,Nt)];
                                       prop_prod_im -= Q01_re[Q_index(t,jCprime,jSprime,kCprime,kSprime,x1,iC,iS,y ,Nc,Ns,Vsrc,Vsnk)] * prop_im[prop_index(2,t,iC,iS,iCprime,iSprime,y,x2 ,Nc,Ns,Vsrc,Vsnk,Nt)] + Q01_im[Q_index(t,jCprime,jSprime,kCprime,kSprime,x1,iC,iS,y ,Nc,Ns,Vsrc,Vsnk)] * prop_re[prop_index(2,t,iC,iS,iCprime,iSprime,y,x2 ,Nc,Ns,Vsrc,Vsnk,Nt)];
                                       for (n=0; n<Nsrc; n++) {
                                          Bdouble_re[Bdouble_index(t,iCprime,iSprime,kCprime,kSprime,x1,jCprime,jSprime,x2,n ,Nc,Ns,Vsnk,Nsrc)] += psi_re[index_2d(y,n ,Nsrc)] * prop_prod_re - psi_im[index_2d(y,n ,Nsrc)] * prop_prod_im;
                                          Bdouble_im[Bdouble_index(t,iCprime,iSprime,kCprime,kSprime,x1,jCprime,jSprime,x2,n ,Nc,Ns,Vsnk,Nsrc)] += psi_re[index_2d(y,n ,Nsrc)] * prop_prod_im + psi_im[index_2d(y,n ,Nsrc)] * prop_prod_re;
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
         }
      }
   }
   free(Q12_re);
   free(Q12_im);
   free(Q01_re);
   free(Q01_im);
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
    const int Nc,
    const int Ns,
    const int Vsrc,
    const int Vsnk,
    const int Nt,
    const int Nw,
    const int Nq,
    const int Nsrc,
    const int Nsnk,
    const int Nperms) {
   /* indices */
   int iC1,iS1,jC1,jS1,kC1,kS1,iC2,iS2,jC2,jS2,kC2,kS2,x1,x2,t,wnum,nperm,b,n,m;
   int Nb = 2;
   int Nw2 = Nw*Nw;
   double term_re, term_im, new_term_re, new_term_im;
   /* build dibaryon */
   int snk_1_nq[Nb];
   int snk_2_nq[Nb];
   int snk_3_nq[Nb];
   int snk_1_b[Nb];
   int snk_2_b[Nb];
   int snk_3_b[Nb];
   int snk_1[Nb];
   int snk_2[Nb];
   int snk_3[Nb];
   for (nperm=0; nperm<Nperms; nperm++) {
      for (b=0; b<Nb; b++) {
         snk_1[b] = perms[index_2d(nperm,Nq*b+0 ,2*Nq)] - 1;
         snk_2[b] = perms[index_2d(nperm,Nq*b+1 ,2*Nq)] - 1;
         snk_3[b] = perms[index_2d(nperm,Nq*b+2 ,2*Nq)] - 1;
         snk_1_b[b] = (snk_1[b] - snk_1[b] % Nq) / Nq;
         snk_2_b[b] = (snk_2[b] - snk_2[b] % Nq) / Nq;
         snk_3_b[b] = (snk_3[b] - snk_3[b] % Nq) / Nq;
         snk_1_nq[b] = snk_1[b] % Nq;
         snk_2_nq[b] = snk_2[b] % Nq;
         snk_3_nq[b] = snk_3[b] % Nq;
      }
      for (wnum=0; wnum< Nw2; wnum++) {
         iC1 = snk_color_weights[index_3d(snk_1_b[0],wnum,snk_1_nq[0] ,Nw2,Nq)];
         iS1 = snk_spin_weights[index_3d(snk_1_b[0],wnum,snk_1_nq[0] ,Nw2,Nq)];
         jC1 = snk_color_weights[index_3d(snk_2_b[0],wnum,snk_2_nq[0] ,Nw2,Nq)];
         jS1 = snk_spin_weights[index_3d(snk_2_b[0],wnum,snk_2_nq[0] ,Nw2,Nq)];
         kC1 = snk_color_weights[index_3d(snk_3_b[0],wnum,snk_3_nq[0] ,Nw2,Nq)];
         kS1 = snk_spin_weights[index_3d(snk_3_b[0],wnum,snk_3_nq[0] ,Nw2,Nq)];
         iC2 = snk_color_weights[index_3d(snk_1_b[1],wnum,snk_1_nq[1] ,Nw2,Nq)];
         iS2 = snk_spin_weights[index_3d(snk_1_b[1],wnum,snk_1_nq[1] ,Nw2,Nq)];
         jC2 = snk_color_weights[index_3d(snk_2_b[1],wnum,snk_2_nq[1] ,Nw2,Nq)];
         jS2 = snk_spin_weights[index_3d(snk_2_b[1],wnum,snk_2_nq[1] ,Nw2,Nq)];
         kC2 = snk_color_weights[index_3d(snk_3_b[1],wnum,snk_3_nq[1] ,Nw2,Nq)];
         kS2 = snk_spin_weights[index_3d(snk_3_b[1],wnum,snk_3_nq[1] ,Nw2,Nq)]; 
         for (t=0; t<Nt; t++) {
            for (x1=0; x1<Vsnk; x1++) {
               for (x2=0; x2<Vsnk; x2++) {
                  for (m=0; m<Nsrc; m++) {
                     term_re = sigs[nperm] * overall_weight * snk_weights[wnum];
                     term_im = 0.0;
                     for (b=0; b<Nb; b++) {
                        if ((snk_1_b[b] == 0) && (snk_2_b[b] == 0) && (snk_3_b[b] == 0)) {
                           new_term_re = term_re * B1_Blocal_re[Blocal_index(t,iC1,iS1,kC1,kS1,x1,jC1,jS1,m ,Nc,Ns,Vsnk,Nsrc)] - term_im * B1_Blocal_im[Blocal_index(t,iC1,iS1,kC1,kS1,x1,jC1,jS1,m ,Nc,Ns,Vsnk,Nsrc)];
                           new_term_im = term_re * B1_Blocal_im[Blocal_index(t,iC1,iS1,kC1,kS1,x1,jC1,jS1,m ,Nc,Ns,Vsnk,Nsrc)] + term_im * B1_Blocal_re[Blocal_index(t,iC1,iS1,kC1,kS1,x1,jC1,jS1,m ,Nc,Ns,Vsnk,Nsrc)];
                        }
                        else if ((snk_1_b[b] == 1) && (snk_2_b[b] == 1) && (snk_3_b[b] == 1)) {
                           new_term_re = term_re * B2_Blocal_re[Blocal_index(t,iC2,iS2,kC2,kS2,x2,jC2,jS2,m ,Nc,Ns,Vsnk,Nsrc)] - term_im * B2_Blocal_im[Blocal_index(t,iC2,iS2,kC2,kS2,x2,jC2,jS2,m ,Nc,Ns,Vsnk,Nsrc)];
                           new_term_im = term_re * B2_Blocal_im[Blocal_index(t,iC2,iS2,kC2,kS2,x2,jC2,jS2,m ,Nc,Ns,Vsnk,Nsrc)] + term_im * B2_Blocal_re[Blocal_index(t,iC2,iS2,kC2,kS2,x2,jC2,jS2,m ,Nc,Ns,Vsnk,Nsrc)];
                        }
                        else if ((snk_1_b[b] == 0) && (snk_3_b[b] == 0)) {
                           new_term_re = term_re * B1_Bsingle_re[Bdouble_index(t,iC1,iS1,kC1,kS1,x1,jC1,jS1,x2,m ,Nc,Ns,Vsnk,Nsrc)] - term_im * B1_Bsingle_im[Bdouble_index(t,iC1,iS1,kC1,kS1,x1,jC1,jS1,x2,m ,Nc,Ns,Vsnk,Nsrc)];
                           new_term_im = term_re * B1_Bsingle_im[Bdouble_index(t,iC1,iS1,kC1,kS1,x1,jC1,jS1,x2,m ,Nc,Ns,Vsnk,Nsrc)] + term_im * B1_Bsingle_re[Bdouble_index(t,iC1,iS1,kC1,kS1,x1,jC1,jS1,x2,m ,Nc,Ns,Vsnk,Nsrc)];
                        }
                        else if ((snk_1_b[b] == 1) && (snk_3_b[b] == 1)) {
                           new_term_re = term_re * B2_Bsingle_re[Bdouble_index(t,iC2,iS2,kC2,kS2,x2,jC2,jS2,x1,m ,Nc,Ns,Vsnk,Nsrc)] - term_im * B2_Bsingle_im[Bdouble_index(t,iC2,iS2,kC2,kS2,x2,jC2,jS2,x1,m ,Nc,Ns,Vsnk,Nsrc)];
                           new_term_im = term_re * B2_Bsingle_im[Bdouble_index(t,iC2,iS2,kC2,kS2,x2,jC2,jS2,x1,m ,Nc,Ns,Vsnk,Nsrc)] + term_im * B2_Bsingle_re[Bdouble_index(t,iC2,iS2,kC2,kS2,x2,jC2,jS2,x1,m ,Nc,Ns,Vsnk,Nsrc)];
                        }
                        else if (((snk_1_b[b] == 0) && (snk_2_b[b] == 0)) || ((snk_2_b[b] == 0) && (snk_3_b[b] == 0))) {
                           new_term_re = term_re * B1_Bdouble_re[Bdouble_index(t,iC1,iS1,kC1,kS1,x1,jC1,jS1,x2,m ,Nc,Ns,Vsnk,Nsrc)] - term_im * B1_Bdouble_im[Bdouble_index(t,iC1,iS1,kC1,kS1,x1,jC1,jS1,x2,m ,Nc,Ns,Vsnk,Nsrc)];
                           new_term_im = term_re * B1_Bdouble_im[Bdouble_index(t,iC1,iS1,kC1,kS1,x1,jC1,jS1,x2,m ,Nc,Ns,Vsnk,Nsrc)] + term_im * B1_Bdouble_re[Bdouble_index(t,iC1,iS1,kC1,kS1,x1,jC1,jS1,x2,m ,Nc,Ns,Vsnk,Nsrc)];
                        }
                        else if (((snk_1_b[b] == 1) && (snk_2_b[b] == 1)) || ((snk_2_b[b] == 1) && (snk_3_b[b] == 1))) {
                           new_term_re = term_re * B2_Bdouble_re[Bdouble_index(t,iC2,iS2,kC2,kS2,x2,jC2,jS2,x1,m ,Nc,Ns,Vsnk,Nsrc)] - term_im * B2_Bdouble_im[Bdouble_index(t,iC2,iS2,kC2,kS2,x2,jC2,jS2,x1,m ,Nc,Ns,Vsnk,Nsrc)];
                           new_term_im = term_re * B2_Bdouble_im[Bdouble_index(t,iC2,iS2,kC2,kS2,x2,jC2,jS2,x1,m ,Nc,Ns,Vsnk,Nsrc)] + term_im * B2_Bdouble_re[Bdouble_index(t,iC2,iS2,kC2,kS2,x2,jC2,jS2,x1,m ,Nc,Ns,Vsnk,Nsrc)];
                        }
                        term_re = new_term_re;
                        term_im = new_term_im;
                     }
                     for (n=0; n<Nsnk; n++) {
                        C_re[index_3d(m,n,t,Nsnk,Nt)] += snk_psi_re[index_3d(x1,x2,n ,Vsnk,Nsnk)] * term_re - snk_psi_im[index_3d(x1,x2,n ,Vsnk,Nsnk)] * term_im;
                        C_im[index_3d(m,n,t,Nsnk,Nt)] += snk_psi_re[index_3d(x1,x2,n ,Vsnk,Nsnk)] * term_im + snk_psi_im[index_3d(x1,x2,n ,Vsnk,Nsnk)] * term_re;
                     }
                  }
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
    const int Nc,
    const int Ns,
    const int Vsrc,
    const int Vsnk,
    const int Nt,
    const int Nw,
    const int Nq,
    const int Nsrc,
    const int NsnkHex,
    const int Nperms) {
   /* indices */
   int iC1,iS1,jC1,jS1,kC1,kS1,iC2,iS2,jC2,jS2,kC2,kS2,x,t,wnum,nperm,b,n,m;
   int Nb = 2;
   int Nw2 = Nw*Nw;
   double term_re, term_im;
   /* build dibaryon */
   int snk_1_nq[Nb];
   int snk_2_nq[Nb];
   int snk_3_nq[Nb];
   int snk_1_b[Nb];
   int snk_2_b[Nb];
   int snk_3_b[Nb];
   int snk_1[Nb];
   int snk_2[Nb];
   int snk_3[Nb];
   for (nperm=0; nperm<Nperms; nperm++) {
      for (b=0; b<Nb; b++) {
         snk_1[b] = perms[index_2d(nperm,Nq*b+0 ,2*Nq)] - 1;
         snk_2[b] = perms[index_2d(nperm,Nq*b+1 ,2*Nq)] - 1;
         snk_3[b] = perms[index_2d(nperm,Nq*b+2 ,2*Nq)] - 1;
         snk_1_b[b] = (snk_1[b] - snk_1[b] % Nq) / Nq;
         snk_2_b[b] = (snk_2[b] - snk_2[b] % Nq) / Nq;
         snk_3_b[b] = (snk_3[b] - snk_3[b] % Nq) / Nq;
         snk_1_nq[b] = snk_1[b] % Nq;
         snk_2_nq[b] = snk_2[b] % Nq;
         snk_3_nq[b] = snk_3[b] % Nq;
      }
      for (wnum=0; wnum< Nw2; wnum++) {
         iC1 = snk_color_weights[index_3d(snk_1_b[0],wnum,snk_1_nq[0] ,Nw2,Nq)];
         iS1 = snk_spin_weights[index_3d(snk_1_b[0],wnum,snk_1_nq[0] ,Nw2,Nq)];
         jC1 = snk_color_weights[index_3d(snk_2_b[0],wnum,snk_2_nq[0] ,Nw2,Nq)];
         jS1 = snk_spin_weights[index_3d(snk_2_b[0],wnum,snk_2_nq[0] ,Nw2,Nq)];
         kC1 = snk_color_weights[index_3d(snk_3_b[0],wnum,snk_3_nq[0] ,Nw2,Nq)];
         kS1 = snk_spin_weights[index_3d(snk_3_b[0],wnum,snk_3_nq[0] ,Nw2,Nq)];
         iC2 = snk_color_weights[index_3d(snk_1_b[1],wnum,snk_1_nq[1] ,Nw2,Nq)];
         iS2 = snk_spin_weights[index_3d(snk_1_b[1],wnum,snk_1_nq[1] ,Nw2,Nq)];
         jC2 = snk_color_weights[index_3d(snk_2_b[1],wnum,snk_2_nq[1] ,Nw2,Nq)];
         jS2 = snk_spin_weights[index_3d(snk_2_b[1],wnum,snk_2_nq[1] ,Nw2,Nq)];
         kC2 = snk_color_weights[index_3d(snk_3_b[1],wnum,snk_3_nq[1] ,Nw2,Nq)];
         kS2 = snk_spin_weights[index_3d(snk_3_b[1],wnum,snk_3_nq[1] ,Nw2,Nq)]; 
         for (t=0; t<Nt; t++) {
            for (x=0; x<Vsnk; x++) {
               for (m=0; m<Nsrc; m++) {
                  term_re = sigs[nperm] * overall_weight * snk_weights[wnum] * (B1_Blocal_re[Blocal_index(t,iC1,iS1,kC1,kS1,x,jC1,jS1,m ,Nc,Ns,Vsnk,Nsrc)] * B2_Blocal_re[Blocal_index(t,iC2,iS2,kC2,kS2,x,jC2,jS2,m ,Nc,Ns,Vsnk,Nsrc)] - B1_Blocal_im[Blocal_index(t,iC1,iS1,kC1,kS1,x,jC1,jS1,m ,Nc,Ns,Vsnk,Nsrc)] * B2_Blocal_im[Blocal_index(t,iC2,iS2,kC2,kS2,x,jC2,jS2,m ,Nc,Ns,Vsnk,Nsrc)]);
                  term_im = sigs[nperm] * overall_weight * snk_weights[wnum] * (B1_Blocal_re[Blocal_index(t,iC1,iS1,kC1,kS1,x,jC1,jS1,m ,Nc,Ns,Vsnk,Nsrc)] * B2_Blocal_im[Blocal_index(t,iC2,iS2,kC2,kS2,x,jC2,jS2,m ,Nc,Ns,Vsnk,Nsrc)] + B1_Blocal_im[Blocal_index(t,iC1,iS1,kC1,kS1,x,jC1,jS1,m ,Nc,Ns,Vsnk,Nsrc)] * B2_Blocal_re[Blocal_index(t,iC2,iS2,kC2,kS2,x,jC2,jS2,m ,Nc,Ns,Vsnk,Nsrc)]);
                  for (n=0; n<NsnkHex; n++) {
                     C_re[index_3d(m,n,t ,NsnkHex,Nt)] += hex_snk_psi_re[index_2d(x,n ,NsnkHex)] * term_re - hex_snk_psi_im[index_2d(x,n ,NsnkHex)] * term_im;
                     C_im[index_3d(m,n,t ,NsnkHex,Nt)] += hex_snk_psi_re[index_2d(x,n ,NsnkHex)] * term_im + hex_snk_psi_im[index_2d(x,n ,NsnkHex)] * term_re;
                  }
               }
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
    const int Nc,
    const int Ns,
    const int Vsrc,
    const int Vsnk,
    const int Nt,
    const int Nw,
    const int Nq,
    const int NsrcHex,
    const int Nsnk,
    const int Nperms) {
   /* indices */
   int iC1,iS1,jC1,jS1,kC1,kS1,iC2,iS2,jC2,jS2,kC2,kS2,x,t,wnum,nperm,b,n,m,y;
   int Nb = 2;
   int Nw2 = Nw*Nw;
   double term_re, term_im;
   /* build dibaryon */
   int src_1_nq[Nb];
   int src_2_nq[Nb];
   int src_3_nq[Nb];
   int src_1_b[Nb];
   int src_2_b[Nb];
   int src_3_b[Nb];
   int src_1[Nb];
   int src_2[Nb];
   int src_3[Nb];
   for (nperm=0; nperm<Nperms; nperm++) {
      for (b=0; b<Nb; b++) {
         src_1[b] = perms[index_2d(nperm,Nq*b+0 ,2*Nq)] - 1;
         src_2[b] = perms[index_2d(nperm,Nq*b+1 ,2*Nq)] - 1;
         src_3[b] = perms[index_2d(nperm,Nq*b+2 ,2*Nq)] - 1;
         src_1_b[b] = (src_1[b] - src_1[b] % Nq) / Nq;
         src_2_b[b] = (src_2[b] - src_2[b] % Nq) / Nq;
         src_3_b[b] = (src_3[b] - src_3[b] % Nq) / Nq;
         src_1_nq[b] = src_1[b] % Nq;
         src_2_nq[b] = src_2[b] % Nq;
         src_3_nq[b] = src_3[b] % Nq;
      }
      for (wnum=0; wnum< Nw2; wnum++) {
         iC1 = src_color_weights[index_3d(src_1_b[0],wnum,src_1_nq[0] ,Nw2,Nq)];
         iS1 = src_spin_weights[index_3d(src_1_b[0],wnum,src_1_nq[0] ,Nw2,Nq)];
         jC1 = src_color_weights[index_3d(src_2_b[0],wnum,src_2_nq[0] ,Nw2,Nq)];
         jS1 = src_spin_weights[index_3d(src_2_b[0],wnum,src_2_nq[0] ,Nw2,Nq)];
         kC1 = src_color_weights[index_3d(src_3_b[0],wnum,src_3_nq[0] ,Nw2,Nq)];
         kS1 = src_spin_weights[index_3d(src_3_b[0],wnum,src_3_nq[0] ,Nw2,Nq)];
         iC2 = src_color_weights[index_3d(src_1_b[1],wnum,src_1_nq[1] ,Nw2,Nq)];
         iS2 = src_spin_weights[index_3d(src_1_b[1],wnum,src_1_nq[1] ,Nw2,Nq)];
         jC2 = src_color_weights[index_3d(src_2_b[1],wnum,src_2_nq[1] ,Nw2,Nq)];
         jS2 = src_spin_weights[index_3d(src_2_b[1],wnum,src_2_nq[1] ,Nw2,Nq)];
         kC2 = src_color_weights[index_3d(src_3_b[1],wnum,src_3_nq[1] ,Nw2,Nq)];
         kS2 = src_spin_weights[index_3d(src_3_b[1],wnum,src_3_nq[1] ,Nw2,Nq)]; 
         for (t=0; t<Nt; t++) {
            for (y=0; y<Vsrc; y++) {
               for (n=0; n<Nsnk; n++) {
                  term_re = sigs[nperm] * overall_weight * src_weights[wnum] * (B1_Blocal_re[Blocal_index(t,iC1,iS1,kC1,kS1,y,jC1,jS1,n ,Nc,Ns,Vsrc,Nsnk)] * B2_Blocal_re[Blocal_index(t,iC2,iS2,kC2,kS2,y,jC2,jS2,n ,Nc,Ns,Vsrc,Nsnk)] - B1_Blocal_im[Blocal_index(t,iC1,iS1,kC1,kS1,y,jC1,jS1,n ,Nc,Ns,Vsrc,Nsnk)] * B2_Blocal_im[Blocal_index(t,iC2,iS2,kC2,kS2,y,jC2,jS2,n ,Nc,Ns,Vsrc,Nsnk)]);
                  term_im = sigs[nperm] * overall_weight * src_weights[wnum] * (B1_Blocal_re[Blocal_index(t,iC1,iS1,kC1,kS1,y,jC1,jS1,n ,Nc,Ns,Vsrc,Nsnk)] * B2_Blocal_im[Blocal_index(t,iC2,iS2,kC2,kS2,y,jC2,jS2,n ,Nc,Ns,Vsrc,Nsnk)] + B1_Blocal_im[Blocal_index(t,iC1,iS1,kC1,kS1,y,jC1,jS1,n ,Nc,Ns,Vsrc,Nsnk)] * B2_Blocal_re[Blocal_index(t,iC2,iS2,kC2,kS2,y,jC2,jS2,n ,Nc,Ns,Vsrc,Nsnk)]);
                  for (m=0; m<NsrcHex; m++) {
                     C_re[index_3d(m,n,t ,Nsnk,Nt)] += hex_src_psi_re[index_2d(y,m, NsrcHex)] * term_re - hex_src_psi_im[index_2d(y,m, NsrcHex)] * term_im;
                     C_im[index_3d(m,n,t ,Nsnk,Nt)] += hex_src_psi_re[index_2d(y,m, NsrcHex)] * term_im + hex_src_psi_im[index_2d(y,m, NsrcHex)] * term_re;
                  }
               }
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
    const int Nc,
    const int Ns,
    const int Vsrc,
    const int Vsnk,
    const int Nt,
    const int Nw,
    const int Nq,
    const int NsrcHex,
    const int NsnkHex,
    const int Nperms) {
   /* indices */
   int x,t,wnum,nperm,b,n,m,y,wnumprime;
   int iC1prime,iS1prime,jC1prime,jS1prime,kC1prime,kS1prime,iC2prime,iS2prime,jC2prime,jS2prime,kC2prime,kS2prime;
   int iC1,iS1,jC1,jS1,kC1,kS1,iC2,iS2,jC2,jS2,kC2,kS2;
   int Nb = 2;
   int Nw2 = Nw*Nw;
   double B1_prop_prod_02_re, B1_prop_prod_02_im, B1_prop_prod_re, B1_prop_prod_im, B2_prop_prod_02_re, B2_prop_prod_02_im, B2_prop_prod_re, B2_prop_prod_im;
   double prop_prod_re, prop_prod_im, new_prop_prod_re, new_prop_prod_im;
   /* build dibaryon */
   int snk_1_nq[Nb];
   int snk_2_nq[Nb];
   int snk_3_nq[Nb];
   int snk_1_b[Nb];
   int snk_2_b[Nb];
   int snk_3_b[Nb];
   int snk_1[Nb];
   int snk_2[Nb];
   int snk_3[Nb];
   for (nperm=0; nperm<Nperms; nperm++) {
      for (b=0; b<Nb; b++) {
         snk_1[b] = perms[index_2d(nperm,Nq*b+0 ,2*Nq)] - 1;
         snk_2[b] = perms[index_2d(nperm,Nq*b+1 ,2*Nq)] - 1;
         snk_3[b] = perms[index_2d(nperm,Nq*b+2 ,2*Nq)] - 1;
         snk_1_b[b] = (snk_1[b] - snk_1[b] % Nq) / Nq;
         snk_2_b[b] = (snk_2[b] - snk_2[b] % Nq) / Nq;
         snk_3_b[b] = (snk_3[b] - snk_3[b] % Nq) / Nq;
         snk_1_nq[b] = snk_1[b] % Nq;
         snk_2_nq[b] = snk_2[b] % Nq;
         snk_3_nq[b] = snk_3[b] % Nq;
      }
      for (wnumprime=0; wnumprime< Nw2; wnumprime++) {
         iC1prime = snk_color_weights[index_3d(snk_1_b[0],wnumprime,snk_1_nq[0] ,Nw2,Nq)];
         iS1prime = snk_spin_weights[index_3d(snk_1_b[0],wnumprime,snk_1_nq[0] ,Nw2,Nq)];
         jC1prime = snk_color_weights[index_3d(snk_2_b[0],wnumprime,snk_2_nq[0] ,Nw2,Nq)];
         jS1prime = snk_spin_weights[index_3d(snk_2_b[0],wnumprime,snk_2_nq[0] ,Nw2,Nq)];
         kC1prime = snk_color_weights[index_3d(snk_3_b[0],wnumprime,snk_3_nq[0] ,Nw2,Nq)];
         kS1prime = snk_spin_weights[index_3d(snk_3_b[0],wnumprime,snk_3_nq[0] ,Nw2,Nq)];
         iC2prime = snk_color_weights[index_3d(snk_1_b[1],wnumprime,snk_1_nq[1] ,Nw2,Nq)];
         iS2prime = snk_spin_weights[index_3d(snk_1_b[1],wnumprime,snk_1_nq[1] ,Nw2,Nq)];
         jC2prime = snk_color_weights[index_3d(snk_2_b[1],wnumprime,snk_2_nq[1] ,Nw2,Nq)];
         jS2prime = snk_spin_weights[index_3d(snk_2_b[1],wnumprime,snk_2_nq[1] ,Nw2,Nq)];
         kC2prime = snk_color_weights[index_3d(snk_3_b[1],wnumprime,snk_3_nq[1] ,Nw2,Nq)];
         kS2prime = snk_spin_weights[index_3d(snk_3_b[1],wnumprime,snk_3_nq[1] ,Nw2,Nq)]; 
         for (t=0; t<Nt; t++) {
            for (y=0; y<Vsrc; y++) {
               for (x=0; x<Vsnk; x++) {
                  B1_prop_prod_re = 0;
                  B1_prop_prod_im = 0;
                  for (wnum=0; wnum<Nw; wnum++) {
                     iC1 = B1_src_color_weights[index_2d(wnum,0, Nq)];
                     iS1 = B1_src_spin_weights[index_2d(wnum,0, Nq)];
                     jC1 = B1_src_color_weights[index_2d(wnum,1, Nq)];
                     jS1 = B1_src_spin_weights[index_2d(wnum,1, Nq)];
                     kC1 = B1_src_color_weights[index_2d(wnum,2, Nq)];
                     kS1 = B1_src_spin_weights[index_2d(wnum,2, Nq)];
                     B1_prop_prod_02_re = B1_src_weights[wnum] * ( (B1_prop_re[prop_index(0,t,iC1,iS1,iC1prime,iS1prime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] * B1_prop_re[prop_index(2,t,kC1,kS1,kC1prime,kS1prime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] - B1_prop_im[prop_index(0,t,iC1,iS1,iC1prime,iS1prime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] * B1_prop_im[prop_index(2,t,kC1,kS1,kC1prime,kS1prime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)]) - (B1_prop_re[prop_index(0,t,iC1,iS1,kC1prime,kS1prime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] * B1_prop_re[prop_index(2,t,kC1,kS1,iC1prime,iS1prime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] - B1_prop_im[prop_index(0,t,iC1,iS1,kC1prime,kS1prime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] * B1_prop_im[prop_index(2,t,kC1,kS1,iC1prime,iS1prime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)]) );
                     B1_prop_prod_02_im = B1_src_weights[wnum] * ( (B1_prop_re[prop_index(0,t,iC1,iS1,iC1prime,iS1prime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] * B1_prop_im[prop_index(2,t,kC1,kS1,kC1prime,kS1prime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] + B1_prop_im[prop_index(0,t,iC1,iS1,iC1prime,iS1prime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] * B1_prop_re[prop_index(2,t,kC1,kS1,kC1prime,kS1prime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)]) - (B1_prop_re[prop_index(0,t,iC1,iS1,kC1prime,kS1prime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] * B1_prop_im[prop_index(2,t,kC1,kS1,iC1prime,iS1prime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] + B1_prop_im[prop_index(0,t,iC1,iS1,kC1prime,kS1prime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] * B1_prop_re[prop_index(2,t,kC1,kS1,iC1prime,iS1prime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)]) );
                     B1_prop_prod_re += B1_prop_prod_02_re * B1_prop_re[prop_index(1,t,jC1,jS1,jC1prime,jS1prime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] - B1_prop_prod_02_im * B1_prop_im[prop_index(1,t,jC1,jS1,jC1prime,jS1prime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)];
                     B1_prop_prod_im += B1_prop_prod_02_re * B1_prop_im[prop_index(1,t,jC1,jS1,jC1prime,jS1prime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] + B1_prop_prod_02_im * B1_prop_re[prop_index(1,t,jC1,jS1,jC1prime,jS1prime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)];
                  }
                  B2_prop_prod_re = 0;
                  B2_prop_prod_im = 0;
                  for (wnum=0; wnum<Nw; wnum++) {
                     iC2 = B2_src_color_weights[index_2d(wnum,0, Nq)];
                     iS2 = B2_src_spin_weights[index_2d(wnum,0, Nq)];
                     jC2 = B2_src_color_weights[index_2d(wnum,1, Nq)];
                     jS2 = B2_src_spin_weights[index_2d(wnum,1, Nq)];
                     kC2 = B2_src_color_weights[index_2d(wnum,2, Nq)];
                     kS2 = B2_src_spin_weights[index_2d(wnum,2, Nq)];
                     B2_prop_prod_02_re = B2_src_weights[wnum] * ( (B2_prop_re[prop_index(0,t,iC2,iS2,iC2prime,iS2prime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] * B2_prop_re[prop_index(2,t,kC2,kS2,kC2prime,kS2prime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] - B2_prop_im[prop_index(0,t,iC2,iS2,iC2prime,iS2prime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] * B2_prop_im[prop_index(2,t,kC2,kS2,kC2prime,kS2prime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)]) - (B2_prop_re[prop_index(0,t,iC2,iS2,kC2prime,kS2prime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] * B2_prop_re[prop_index(2,t,kC2,kS2,iC2prime,iS2prime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] - B2_prop_im[prop_index(0,t,iC2,iS2,kC2prime,kS2prime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] * B2_prop_im[prop_index(2,t,kC2,kS2,iC2prime,iS2prime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)]) );
                     B2_prop_prod_02_im = B2_src_weights[wnum] * ( (B2_prop_re[prop_index(0,t,iC2,iS2,iC2prime,iS2prime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] * B2_prop_im[prop_index(2,t,kC2,kS2,kC2prime,kS2prime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] + B2_prop_im[prop_index(0,t,iC2,iS2,iC2prime,iS2prime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] * B2_prop_re[prop_index(2,t,kC2,kS2,kC2prime,kS2prime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)]) - (B2_prop_re[prop_index(0,t,iC2,iS2,kC2prime,kS2prime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] * B2_prop_im[prop_index(2,t,kC2,kS2,iC2prime,iS2prime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] + B2_prop_im[prop_index(0,t,iC2,iS2,kC2prime,kS2prime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] * B2_prop_re[prop_index(2,t,kC2,kS2,iC2prime,iS2prime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)]) );
                     B2_prop_prod_re += B2_prop_prod_02_re * B2_prop_re[prop_index(1,t,jC2,jS2,jC2prime,jS2prime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] - B2_prop_prod_02_im * B2_prop_im[prop_index(1,t,jC2,jS2,jC2prime,jS2prime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)];
                     B2_prop_prod_im += B2_prop_prod_02_re * B2_prop_im[prop_index(1,t,jC2,jS2,jC2prime,jS2prime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)] + B2_prop_prod_02_im * B2_prop_re[prop_index(1,t,jC2,jS2,jC2prime,jS2prime,y,x ,Nc,Ns,Vsrc,Vsnk,Nt)];
                  }
                  prop_prod_re = overall_weight * sigs[nperm] * snk_weights[wnumprime] * ( B1_prop_prod_re * B2_prop_prod_re - B1_prop_prod_im * B2_prop_prod_im );
                  prop_prod_im = overall_weight * sigs[nperm] * snk_weights[wnumprime] * ( B1_prop_prod_re * B2_prop_prod_im + B1_prop_prod_im * B2_prop_prod_re );
                  for (m=0; m<NsrcHex; m++) {
                     new_prop_prod_re = hex_src_psi_re[index_2d(y,m ,NsrcHex)] * prop_prod_re - hex_src_psi_im[index_2d(y,m ,NsrcHex)] * prop_prod_im;
                     new_prop_prod_im = hex_src_psi_re[index_2d(y,m ,NsrcHex)] * prop_prod_im + hex_src_psi_im[index_2d(y,m ,NsrcHex)] * prop_prod_re;
                     for (n=0; n<NsnkHex; n++) {
                        C_re[index_3d(m,n,t ,NsnkHex,Nt)] += hex_snk_psi_re[index_2d(x,n ,NsnkHex)] * new_prop_prod_re - hex_snk_psi_im[index_2d(x,n ,NsnkHex)] * new_prop_prod_im;
                        C_im[index_3d(m,n,t ,NsnkHex,Nt)] += hex_snk_psi_re[index_2d(x,n ,NsnkHex)] * new_prop_prod_im + hex_snk_psi_im[index_2d(x,n ,NsnkHex)] * new_prop_prod_re;
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
    const int Nc,
    const int Ns,
    const int Vsrc,
    const int Vsnk,
    const int Nt,
    const int Nw,
    const int Nq,
    const int Nsrc,
    const int Nsnk,
    const int NsrcHex,
    const int NsnkHex,
    const int Nperms) {
   /* indices */
   int nB1, nB2, nq, n, m, t;

   double total_time_reference = 0;
   double total_time_tiramisu = 0;
   double total_time_common = 0;

   // hold results for two nucleon correlators 
   /* compute two nucleon snk weights */
   int Nw2 = Nw*Nw;
   int* snk_color_weights_1 = malloc(2 * Nw2 * Nq * sizeof (int));
   int* snk_color_weights_2 = malloc(2 * Nw2 * Nq * sizeof (int));
   int* snk_color_weights_r1 = malloc(2 * Nw2 * Nq * sizeof (int));
   int* snk_color_weights_r2_1 = malloc(2 * Nw2 * Nq * sizeof (int));
   int* snk_color_weights_r2_2 = malloc(2 * Nw2 * Nq * sizeof (int));
   int* snk_color_weights_r3 = malloc(2 * Nw2 * Nq * sizeof (int));
   int* snk_spin_weights_1 = malloc(2 * Nw2 * Nq * sizeof (int));
   int* snk_spin_weights_2 = malloc(2 * Nw2 * Nq * sizeof (int));
   int* snk_spin_weights_r1 = malloc(2 * Nw2 * Nq * sizeof (int));
   int* snk_spin_weights_r2_1 = malloc(2 * Nw2 * Nq * sizeof (int));
   int* snk_spin_weights_r2_2 = malloc(2 * Nw2 * Nq * sizeof (int));
   int* snk_spin_weights_r3 = malloc(2 * Nw2 * Nq * sizeof (int));
   double* snk_weights_1 = malloc(Nw2 * sizeof (double));
   double* snk_weights_2 = malloc(Nw2 * sizeof (double));
   double* snk_weights_r1 = malloc(Nw2 * sizeof (double));
   double* snk_weights_r2_1 = malloc(Nw2 * sizeof (double));
   double* snk_weights_r2_2 = malloc(Nw2 * sizeof (double));
   double* snk_weights_r3 = malloc(Nw2 * sizeof (double));
   for (nB1=0; nB1<Nw; nB1++) {
      for (nB2=0; nB2<Nw; nB2++) {
         snk_weights_1[nB1+Nw*nB2] = 1.0/sqrt(2) * src_weights_r1[nB1]*src_weights_r2[nB2];
         snk_weights_2[nB1+Nw*nB2] = -1.0/sqrt(2) * src_weights_r2[nB1]*src_weights_r1[nB2];
         snk_weights_r1[nB1+Nw*nB2] = src_weights_r1[nB1]*src_weights_r1[nB2];
         snk_weights_r2_1[nB1+Nw*nB2] = 1.0/sqrt(2) * src_weights_r1[nB1]*src_weights_r2[nB2];
         snk_weights_r2_2[nB1+Nw*nB2] = 1.0/sqrt(2) * src_weights_r2[nB1]*src_weights_r1[nB2];
         snk_weights_r3[nB1+Nw*nB2] = src_weights_r2[nB1]*src_weights_r2[nB2];
         for (nq=0; nq<Nq; nq++) {
            // A1g
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
            snk_spin_weights_r3[index_3d(1,nB1+Nw*nB2,nq ,Nw2,Nq)] = src_spin_weights_r2[index_2d(nB2,nq ,Nq)];
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
   if (Nsrc > 0) {
      // create blocks
      double* B1_Blocal_r1_re;
      double* B1_Blocal_r1_im;
      double* B1_Blocal_r2_re;
      double* B1_Blocal_r2_im;
      double* B2_Blocal_r1_re;
      double* B2_Blocal_r1_im;
      double* B2_Blocal_r2_re;
      double* B2_Blocal_r2_im;

      double* t_B1_Blocal_r1_re;
      double* t_B1_Blocal_r1_im;
      double* t_B1_Blocal_r2_re;
      double* t_B1_Blocal_r2_im;
      double* t_B2_Blocal_r1_re;
      double* t_B2_Blocal_r1_im;
      double* t_B2_Blocal_r2_re;
      double* t_B2_Blocal_r2_im;

      if (USE_REFERENCE)
      {
	      B1_Blocal_r1_re = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Nsrc * sizeof (double));
	      B1_Blocal_r1_im = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Nsrc * sizeof (double));
	      B1_Blocal_r2_re = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Nsrc * sizeof (double));
	      B1_Blocal_r2_im = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Nsrc * sizeof (double));
	      B2_Blocal_r1_re = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Nsrc * sizeof (double));
	      B2_Blocal_r1_im = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Nsrc * sizeof (double));
	      B2_Blocal_r2_re = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Nsrc * sizeof (double));
	      B2_Blocal_r2_im = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Nsrc * sizeof (double));

	      double start_time = rtclock();
	      make_local_block(B1_Blocal_r1_re, B1_Blocal_r1_im, B1_prop_re, B1_prop_im, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, src_psi_B1_re, src_psi_B1_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,Nsrc);
	      make_local_block(B1_Blocal_r2_re, B1_Blocal_r2_im, B1_prop_re, B1_prop_im, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, src_psi_B1_re, src_psi_B1_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,Nsrc);
	      make_local_block(B2_Blocal_r1_re, B2_Blocal_r1_im, B2_prop_re, B2_prop_im, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, src_psi_B2_re, src_psi_B2_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,Nsrc);
	      make_local_block(B2_Blocal_r2_re, B2_Blocal_r2_im, B2_prop_re, B2_prop_im, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, src_psi_B2_re, src_psi_B2_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,Nsrc);
	      double end_time = rtclock();
	      total_time_reference += (end_time - start_time);

	      printf("made local blocks \n");
      }
      if (Nsnk > 0) {
         /* BB_BB */
         double* B1_Bsingle_r1_re;
         double* B1_Bsingle_r1_im;
         double* B1_Bsingle_r2_re;
         double* B1_Bsingle_r2_im;
         double* B1_Bdouble_r1_re;
         double* B1_Bdouble_r1_im;
         double* B1_Bdouble_r2_re;
         double* B1_Bdouble_r2_im;
         double* B2_Bsingle_r1_re;
         double* B2_Bsingle_r1_im;
         double* B2_Bsingle_r2_re;
         double* B2_Bsingle_r2_im;
         double* B2_Bdouble_r1_re;
         double* B2_Bdouble_r1_im;
         double* B2_Bdouble_r2_re;
         double* B2_Bdouble_r2_im;

	 double* t_B1_Bsingle_r1_re;
         double* t_B1_Bsingle_r1_im;
         double* t_B1_Bsingle_r2_re;
         double* t_B1_Bsingle_r2_im;
         double* t_B1_Bdouble_r1_re;
         double* t_B1_Bdouble_r1_im;
         double* t_B1_Bdouble_r2_re;
         double* t_B1_Bdouble_r2_im;
         double* t_B2_Bsingle_r1_re;
         double* t_B2_Bsingle_r1_im;
         double* t_B2_Bsingle_r2_re;
         double* t_B2_Bsingle_r2_im;
         double* t_B2_Bdouble_r1_re;
         double* t_B2_Bdouble_r1_im;
         double* t_B2_Bdouble_r2_re;
         double* t_B2_Bdouble_r2_im;

	 if (USE_REFERENCE)
	 {
		 B1_Bsingle_r1_re = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Vsnk * Nsrc * sizeof (double));
		 B1_Bsingle_r1_im = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Vsnk * Nsrc * sizeof (double));
		 B1_Bsingle_r2_re = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Vsnk * Nsrc * sizeof (double));
		 B1_Bsingle_r2_im = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Vsnk * Nsrc * sizeof (double));
		 B1_Bdouble_r1_re = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Vsnk * Nsrc * sizeof (double));
		 B1_Bdouble_r1_im = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Vsnk * Nsrc * sizeof (double));
		 B1_Bdouble_r2_re = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Vsnk * Nsrc * sizeof (double));
		 B1_Bdouble_r2_im = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Vsnk * Nsrc * sizeof (double));
		 B2_Bsingle_r1_re = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Vsnk * Nsrc * sizeof (double));
		 B2_Bsingle_r1_im = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Vsnk * Nsrc * sizeof (double));
		 B2_Bsingle_r2_re = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Vsnk * Nsrc * sizeof (double));
		 B2_Bsingle_r2_im = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Vsnk * Nsrc * sizeof (double));
		 B2_Bdouble_r1_re = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Vsnk * Nsrc * sizeof (double));
		 B2_Bdouble_r1_im = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Vsnk * Nsrc * sizeof (double));
		 B2_Bdouble_r2_re = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Vsnk * Nsrc * sizeof (double));
		 B2_Bdouble_r2_im = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Vsnk * Nsrc * sizeof (double));

   		 double start_time = rtclock();
		 make_single_block(B1_Bsingle_r1_re, B1_Bsingle_r1_im, B1_prop_re, B1_prop_im, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, src_psi_B1_re, src_psi_B1_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,Nsrc);
		 make_single_block(B1_Bsingle_r2_re, B1_Bsingle_r2_im, B1_prop_re, B1_prop_im, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, src_psi_B1_re, src_psi_B1_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,Nsrc);
		 make_double_block(B1_Bdouble_r1_re, B1_Bdouble_r1_im, B1_prop_re, B1_prop_im, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, src_psi_B1_re, src_psi_B1_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,Nsrc);
		 make_double_block(B1_Bdouble_r2_re, B1_Bdouble_r2_im, B1_prop_re, B1_prop_im, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, src_psi_B1_re, src_psi_B1_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,Nsrc);
		 make_single_block(B2_Bsingle_r1_re, B2_Bsingle_r1_im, B2_prop_re, B2_prop_im, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, src_psi_B2_re, src_psi_B2_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,Nsrc);
		 make_single_block(B2_Bsingle_r2_re, B2_Bsingle_r2_im, B2_prop_re, B2_prop_im, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, src_psi_B2_re, src_psi_B2_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,Nsrc);
		 make_double_block(B2_Bdouble_r1_re, B2_Bdouble_r1_im, B2_prop_re, B2_prop_im, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, src_psi_B2_re, src_psi_B2_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,Nsrc);
		 make_double_block(B2_Bdouble_r2_re, B2_Bdouble_r2_im, B2_prop_re, B2_prop_im, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, src_psi_B2_re, src_psi_B2_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,Nsrc);
		 double end_time = rtclock();
		 total_time_reference += (end_time - start_time);

		 printf("made double blocks \n");
	 }
	 if (USE_TIRAMISU)
	 {
		 t_B1_Blocal_r1_re = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Nsrc * sizeof (double));
		 t_B1_Blocal_r1_im = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Nsrc * sizeof (double));
		 t_B1_Blocal_r2_re = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Nsrc * sizeof (double));
		 t_B1_Blocal_r2_im = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Nsrc * sizeof (double));
		 t_B2_Blocal_r1_re = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Nsrc * sizeof (double));
		 t_B2_Blocal_r1_im = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Nsrc * sizeof (double));
		 t_B2_Blocal_r2_re = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Nsrc * sizeof (double));
		 t_B2_Blocal_r2_im = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Nsrc * sizeof (double));

		 t_B1_Bsingle_r1_re = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Vsnk * Nsrc * sizeof (double));
		 t_B1_Bsingle_r1_im = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Vsnk * Nsrc * sizeof (double));
		 t_B1_Bsingle_r2_re = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Vsnk * Nsrc * sizeof (double));
		 t_B1_Bsingle_r2_im = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Vsnk * Nsrc * sizeof (double));
		 t_B1_Bdouble_r1_re = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Vsnk * Nsrc * sizeof (double));
		 t_B1_Bdouble_r1_im = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Vsnk * Nsrc * sizeof (double));
		 t_B1_Bdouble_r2_re = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Vsnk * Nsrc * sizeof (double));
		 t_B1_Bdouble_r2_im = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Vsnk * Nsrc * sizeof (double));
		 t_B2_Bsingle_r1_re = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Vsnk * Nsrc * sizeof (double));
		 t_B2_Bsingle_r1_im = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Vsnk * Nsrc * sizeof (double));
		 t_B2_Bsingle_r2_re = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Vsnk * Nsrc * sizeof (double));
		 t_B2_Bsingle_r2_im = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Vsnk * Nsrc * sizeof (double));
		 t_B2_Bdouble_r1_re = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Vsnk * Nsrc * sizeof (double));
		 t_B2_Bdouble_r1_im = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Vsnk * Nsrc * sizeof (double));
		 t_B2_Bdouble_r2_re = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Vsnk * Nsrc * sizeof (double));
		 t_B2_Bdouble_r2_im = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Vsnk * Nsrc * sizeof (double));

		 double start_time = rtclock();
		 tiramisu_wrapper_make_local_single_double_block(t_B1_Blocal_r1_re, t_B1_Blocal_r1_im, t_B1_Bsingle_r1_re, t_B1_Bsingle_r1_im, t_B1_Bdouble_r1_re, t_B1_Bdouble_r1_im, B1_prop_re, B1_prop_im, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, src_psi_B1_re, src_psi_B1_im, Nc, Ns, Vsrc, Vsnk, Nt, Nw, Nq, Nsrc);
		 tiramisu_wrapper_make_local_single_double_block(t_B1_Blocal_r2_re, t_B1_Blocal_r2_im, t_B1_Bsingle_r2_re, t_B1_Bsingle_r2_im, t_B1_Bdouble_r2_re, t_B1_Bdouble_r2_im, B1_prop_re, B1_prop_im, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, src_psi_B1_re, src_psi_B1_im, Nc, Ns, Vsrc, Vsnk, Nt, Nw, Nq, Nsrc);
		 tiramisu_wrapper_make_local_single_double_block(t_B2_Blocal_r1_re, t_B2_Blocal_r1_im, t_B2_Bsingle_r1_re, t_B2_Bsingle_r1_im, t_B2_Bdouble_r1_re, t_B2_Bdouble_r1_im, B2_prop_re, B2_prop_im, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, src_psi_B2_re, src_psi_B1_im, Nc, Ns, Vsrc, Vsnk, Nt, Nw, Nq, Nsrc);
		 tiramisu_wrapper_make_local_single_double_block(t_B2_Blocal_r2_re, t_B2_Blocal_r2_im, t_B2_Bsingle_r2_re, t_B2_Bsingle_r2_im, t_B2_Bdouble_r2_re, t_B2_Bdouble_r2_im, B2_prop_re, B2_prop_im, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, src_psi_B2_re, src_psi_B1_im, Nc, Ns, Vsrc, Vsnk, Nt, Nw, Nq, Nsrc);
		 double end_time = rtclock();
		 total_time_tiramisu += (end_time - start_time);
	 }
	 /* Compare the results of Reference and Tiramisu code. */
	 if (USE_REFERENCE && USE_TIRAMISU)
	 {
	       int iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, iC, iS, jC, jS, kC, kS, x, t, y, wnum, m;

	       for (t=0; t<Nt; t++)
		  for (iCprime=0; iCprime<Nc; iCprime++)
		     for (iSprime=0; iSprime<Ns; iSprime++)
			for (kCprime=0; kCprime<Nc; kCprime++)
			   for (kSprime=0; kSprime<Ns; kSprime++)
			      for (x=0; x<Vsnk; x++)
				 for (jCprime=0; jCprime<Nc; jCprime++)
				    for (jSprime=0; jSprime<Ns; jSprime++)
				       for (m=0; m<Nsrc; m++)
				       {
					  if (fabs(B1_Blocal_r1_re[ Blocal_index(t,iCprime,iSprime,kCprime,kSprime,x,jCprime,jSprime,m ,Nc,Ns,Vsnk,Nsrc)] -
					         t_B1_Blocal_r1_re[tBlocal_index(t,iCprime,iSprime,kCprime,kSprime,x,m,jCprime,jSprime ,Nc,Ns,Vsnk,Nsrc)]) >= ERROR_THRESH)
					      error_msg("Results for B1_Blocal_r1_re do not match");
					  if (fabs(B1_Blocal_r1_im[ Blocal_index(t,iCprime,iSprime,kCprime,kSprime,x,jCprime,jSprime,m ,Nc,Ns,Vsnk,Nsrc)] -
					         t_B1_Blocal_r1_im[tBlocal_index(t,iCprime,iSprime,kCprime,kSprime,x,m,jCprime,jSprime ,Nc,Ns,Vsnk,Nsrc)]) >= ERROR_THRESH)
					      error_msg("Results for B1_Blocal_r1_im do not match");

					  if (fabs(B2_Blocal_r1_re[ Blocal_index(t,iCprime,iSprime,kCprime,kSprime,x,jCprime,jSprime,m ,Nc,Ns,Vsnk,Nsrc)] -
					         t_B2_Blocal_r1_re[tBlocal_index(t,iCprime,iSprime,kCprime,kSprime,x,m,jCprime,jSprime ,Nc,Ns,Vsnk,Nsrc)]) >= ERROR_THRESH)
					      error_msg("Results for B2_Blocal_r1_re do not match");
					  if (fabs(B2_Blocal_r1_im[ Blocal_index(t,iCprime,iSprime,kCprime,kSprime,x,jCprime,jSprime,m ,Nc,Ns,Vsnk,Nsrc)] -
					         t_B2_Blocal_r1_im[tBlocal_index(t,iCprime,iSprime,kCprime,kSprime,x,m,jCprime,jSprime ,Nc,Ns,Vsnk,Nsrc)]) >= ERROR_THRESH)
					      error_msg("Results for B2_Blocal_r1_im do not match");
				       }
	 }
         /* compute two nucleon correlators from blocks */
         double* BB_0_re = malloc(Nsrc * Nsnk * Nt * sizeof (double));
         double* BB_0_im = malloc(Nsrc * Nsnk * Nt * sizeof (double));
         double* BB_r1_re = malloc(Nsrc * Nsnk * Nt * sizeof (double));
         double* BB_r1_im = malloc(Nsrc * Nsnk * Nt * sizeof (double));
         double* BB_r2_re = malloc(Nsrc * Nsnk * Nt * sizeof (double));
         double* BB_r2_im = malloc(Nsrc * Nsnk * Nt * sizeof (double));
         double* BB_r3_re = malloc(Nsrc * Nsnk * Nt * sizeof (double));
         double* BB_r3_im = malloc(Nsrc * Nsnk * Nt * sizeof (double));

	 double start_time = rtclock();
         for (m=0; m<Nsrc; m++) {
            for (n=0; n<Nsnk; n++) {
               for (t=0; t<Nt; t++) {
                  BB_0_re[index_3d(m,n,t,Nsnk,Nt)] = 0.0;
                  BB_0_im[index_3d(m,n,t,Nsnk,Nt)] = 0.0;
                  BB_r1_re[index_3d(m,n,t,Nsnk,Nt)] = 0.0;
                  BB_r1_im[index_3d(m,n,t,Nsnk,Nt)] = 0.0;
                  BB_r2_re[index_3d(m,n,t,Nsnk,Nt)] = 0.0;
                  BB_r2_im[index_3d(m,n,t,Nsnk,Nt)] = 0.0;
                  BB_r3_re[index_3d(m,n,t,Nsnk,Nt)] = 0.0;
                  BB_r3_im[index_3d(m,n,t,Nsnk,Nt)] = 0.0;
               }
            }
         }
         make_dibaryon_correlator(BB_0_re, BB_0_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B1_Bsingle_r1_re, B1_Bsingle_r1_im, B1_Bdouble_r1_re, B1_Bdouble_r1_im, B2_Blocal_r2_re, B2_Blocal_r2_im, B2_Bsingle_r2_re, B2_Bsingle_r2_im, B2_Bdouble_r2_re, B2_Bdouble_r2_im, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_1, snk_spin_weights_1, snk_weights_1, snk_psi_re, snk_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,Nsrc,Nsnk,Nperms);
         make_dibaryon_correlator(BB_0_re, BB_0_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B1_Bsingle_r1_re, B1_Bsingle_r1_im, B1_Bdouble_r1_re, B1_Bdouble_r1_im, B2_Blocal_r2_re, B2_Blocal_r2_im, B2_Bsingle_r2_re, B2_Bsingle_r2_im, B2_Bdouble_r2_re, B2_Bdouble_r2_im, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_2, snk_spin_weights_2, snk_weights_2, snk_psi_re, snk_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,Nsrc,Nsnk,Nperms);
         make_dibaryon_correlator(BB_0_re, BB_0_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B1_Bsingle_r2_re, B1_Bsingle_r2_im, B1_Bdouble_r2_re, B1_Bdouble_r2_im, B2_Blocal_r1_re, B2_Blocal_r1_im, B2_Bsingle_r1_re, B2_Bsingle_r1_im, B2_Bdouble_r1_re, B2_Bdouble_r1_im, perms, sigs, -1.0*overall_weight/sqrt(2.0), snk_color_weights_1, snk_spin_weights_1, snk_weights_1, snk_psi_re, snk_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,Nsrc,Nsnk,Nperms);
         make_dibaryon_correlator(BB_0_re, BB_0_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B1_Bsingle_r2_re, B1_Bsingle_r2_im, B1_Bdouble_r2_re, B1_Bdouble_r2_im, B2_Blocal_r1_re, B2_Blocal_r1_im, B2_Bsingle_r1_re, B2_Bsingle_r1_im, B2_Bdouble_r1_re, B2_Bdouble_r1_im, perms, sigs, -1.0*overall_weight/sqrt(2.0), snk_color_weights_2, snk_spin_weights_2, snk_weights_2, snk_psi_re, snk_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,Nsrc,Nsnk,Nperms);
         make_dibaryon_correlator(BB_r1_re, BB_r1_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B1_Bsingle_r1_re, B1_Bsingle_r1_im, B1_Bdouble_r1_re, B1_Bdouble_r1_im, B2_Blocal_r1_re, B2_Blocal_r1_im, B2_Bsingle_r1_re, B2_Bsingle_r1_im, B2_Bdouble_r1_re, B2_Bdouble_r1_im, perms, sigs, overall_weight, snk_color_weights_r1, snk_spin_weights_r1, snk_weights_r1, snk_psi_re, snk_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,Nsrc,Nsnk,Nperms);
         make_dibaryon_correlator(BB_r2_re, BB_r2_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B1_Bsingle_r1_re, B1_Bsingle_r1_im, B1_Bdouble_r1_re, B1_Bdouble_r1_im, B2_Blocal_r2_re, B2_Blocal_r2_im, B2_Bsingle_r2_re, B2_Bsingle_r2_im, B2_Bdouble_r2_re, B2_Bdouble_r2_im, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_r2_1, snk_spin_weights_r2_1, snk_weights_r2_1, snk_psi_re, snk_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,Nsrc,Nsnk,Nperms);
         make_dibaryon_correlator(BB_r2_re, BB_r2_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B1_Bsingle_r1_re, B1_Bsingle_r1_im, B1_Bdouble_r1_re, B1_Bdouble_r1_im, B2_Blocal_r2_re, B2_Blocal_r2_im, B2_Bsingle_r2_re, B2_Bsingle_r2_im, B2_Bdouble_r2_re, B2_Bdouble_r2_im, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_r2_2, snk_spin_weights_r2_2, snk_weights_r2_2, snk_psi_re, snk_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,Nsrc,Nsnk,Nperms);
         make_dibaryon_correlator(BB_r2_re, BB_r2_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B1_Bsingle_r2_re, B1_Bsingle_r2_im, B1_Bdouble_r2_re, B1_Bdouble_r2_im, B2_Blocal_r1_re, B2_Blocal_r1_im, B2_Bsingle_r1_re, B2_Bsingle_r1_im, B2_Bdouble_r1_re, B2_Bdouble_r1_im, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_r2_1, snk_spin_weights_r2_1, snk_weights_r2_1, snk_psi_re, snk_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,Nsrc,Nsnk,Nperms);
         make_dibaryon_correlator(BB_r2_re, BB_r2_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B1_Bsingle_r2_re, B1_Bsingle_r2_im, B1_Bdouble_r2_re, B1_Bdouble_r2_im, B2_Blocal_r1_re, B2_Blocal_r1_im, B2_Bsingle_r1_re, B2_Bsingle_r1_im, B2_Bdouble_r1_re, B2_Bdouble_r1_im, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_r2_2, snk_spin_weights_r2_2, snk_weights_r2_2, snk_psi_re, snk_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,Nsrc,Nsnk,Nperms);
         make_dibaryon_correlator(BB_r3_re, BB_r3_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B1_Bsingle_r2_re, B1_Bsingle_r2_im, B1_Bdouble_r2_re, B1_Bdouble_r2_im, B2_Blocal_r2_re, B2_Blocal_r2_im, B2_Bsingle_r2_re, B2_Bsingle_r2_im, B2_Bdouble_r2_re, B2_Bdouble_r2_im, perms, sigs, overall_weight, snk_color_weights_r3, snk_spin_weights_r3, snk_weights_r3, snk_psi_re, snk_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,Nsrc,Nsnk,Nperms);
         if (space_symmetric != 0) {
            make_dibaryon_correlator(BB_0_re, BB_0_im, B2_Blocal_r1_re, B2_Blocal_r1_im, B2_Bsingle_r1_re, B2_Bsingle_r1_im, B2_Bdouble_r1_re, B2_Bdouble_r1_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B1_Bsingle_r2_re, B1_Bsingle_r2_im, B1_Bdouble_r2_re, B1_Bdouble_r2_im, perms, sigs, space_symmetric*overall_weight/sqrt(2.0), snk_color_weights_1, snk_spin_weights_1, snk_weights_1, snk_psi_re, snk_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,Nsrc,Nsnk,Nperms);
            make_dibaryon_correlator(BB_0_re, BB_0_im, B2_Blocal_r1_re, B2_Blocal_r1_im, B2_Bsingle_r1_re, B2_Bsingle_r1_im, B2_Bdouble_r1_re, B2_Bdouble_r1_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B1_Bsingle_r2_re, B1_Bsingle_r2_im, B1_Bdouble_r2_re, B1_Bdouble_r2_im, perms, sigs, space_symmetric*overall_weight/sqrt(2.0), snk_color_weights_2, snk_spin_weights_2, snk_weights_2, snk_psi_re, snk_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,Nsrc,Nsnk,Nperms);
            make_dibaryon_correlator(BB_0_re, BB_0_im, B2_Blocal_r2_re, B2_Blocal_r2_im, B2_Bsingle_r2_re, B2_Bsingle_r2_im, B2_Bdouble_r2_re, B2_Bdouble_r2_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B1_Bsingle_r1_re, B1_Bsingle_r1_im, B1_Bdouble_r1_re, B1_Bdouble_r1_im, perms, sigs, -1.0*space_symmetric*overall_weight/sqrt(2.0), snk_color_weights_1, snk_spin_weights_1, snk_weights_1, snk_psi_re, snk_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,Nsrc,Nsnk,Nperms);
            make_dibaryon_correlator(BB_0_re, BB_0_im, B2_Blocal_r2_re, B2_Blocal_r2_im, B2_Bsingle_r2_re, B2_Bsingle_r2_im, B2_Bdouble_r2_re, B2_Bdouble_r2_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B1_Bsingle_r1_re, B1_Bsingle_r1_im, B1_Bdouble_r1_re, B1_Bdouble_r1_im, perms, sigs, -1.0*space_symmetric*overall_weight/sqrt(2.0), snk_color_weights_2, snk_spin_weights_2, snk_weights_2, snk_psi_re, snk_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,Nsrc,Nsnk,Nperms);
            make_dibaryon_correlator(BB_r1_re, BB_r1_im, B2_Blocal_r1_re, B2_Blocal_r1_im, B2_Bsingle_r1_re, B2_Bsingle_r1_im, B2_Bdouble_r1_re, B2_Bdouble_r1_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B1_Bsingle_r1_re, B1_Bsingle_r1_im, B1_Bdouble_r1_re, B1_Bdouble_r1_im, perms, sigs, space_symmetric*overall_weight, snk_color_weights_r1, snk_spin_weights_r1, snk_weights_r1, snk_psi_re, snk_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,Nsrc,Nsnk,Nperms);
            make_dibaryon_correlator(BB_r2_re, BB_r2_im, B2_Blocal_r1_re, B2_Blocal_r1_im, B2_Bsingle_r1_re, B2_Bsingle_r1_im, B2_Bdouble_r1_re, B2_Bdouble_r1_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B1_Bsingle_r2_re, B1_Bsingle_r2_im, B1_Bdouble_r2_re, B1_Bdouble_r2_im, perms, sigs, space_symmetric*overall_weight/sqrt(2.0), snk_color_weights_r2_1, snk_spin_weights_r2_1, snk_weights_r2_1, snk_psi_re, snk_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,Nsrc,Nsnk,Nperms);
            make_dibaryon_correlator(BB_r2_re, BB_r2_im, B2_Blocal_r1_re, B2_Blocal_r1_im, B2_Bsingle_r1_re, B2_Bsingle_r1_im, B2_Bdouble_r1_re, B2_Bdouble_r1_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B1_Bsingle_r2_re, B1_Bsingle_r2_im, B1_Bdouble_r2_re, B1_Bdouble_r2_im, perms, sigs, space_symmetric*overall_weight/sqrt(2.0), snk_color_weights_r2_2, snk_spin_weights_r2_2, snk_weights_r2_2, snk_psi_re, snk_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,Nsrc,Nsnk,Nperms);
            make_dibaryon_correlator(BB_r2_re, BB_r2_im, B2_Blocal_r2_re, B2_Blocal_r2_im, B2_Bsingle_r2_re, B2_Bsingle_r2_im, B2_Bdouble_r2_re, B2_Bdouble_r2_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B1_Bsingle_r1_re, B1_Bsingle_r1_im, B1_Bdouble_r1_re, B1_Bdouble_r1_im, perms, sigs, space_symmetric*overall_weight/sqrt(2.0), snk_color_weights_r2_1, snk_spin_weights_r2_1, snk_weights_r2_1, snk_psi_re, snk_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,Nsrc,Nsnk,Nperms);
            make_dibaryon_correlator(BB_r2_re, BB_r2_im, B2_Blocal_r2_re, B2_Blocal_r2_im, B2_Bsingle_r2_re, B2_Bsingle_r2_im, B2_Bdouble_r2_re, B2_Bdouble_r2_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B1_Bsingle_r1_re, B1_Bsingle_r1_im, B1_Bdouble_r1_re, B1_Bdouble_r1_im, perms, sigs, space_symmetric*overall_weight/sqrt(2.0), snk_color_weights_r2_2, snk_spin_weights_r2_2, snk_weights_r2_2, snk_psi_re, snk_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,Nsrc,Nsnk,Nperms);
            make_dibaryon_correlator(BB_r3_re, BB_r3_im, B2_Blocal_r2_re, B2_Blocal_r2_im, B2_Bsingle_r2_re, B2_Bsingle_r2_im, B2_Bdouble_r2_re, B2_Bdouble_r2_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B1_Bsingle_r2_re, B1_Bsingle_r2_im, B1_Bdouble_r2_re, B1_Bdouble_r2_im, perms, sigs, space_symmetric*overall_weight, snk_color_weights_r3, snk_spin_weights_r3, snk_weights_r3, snk_psi_re, snk_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,Nsrc,Nsnk,Nperms);
         }
         for (m=0; m<Nsrc; m++) {
            for (n=0; n<Nsnk; n++) {
               for (t=0; t<Nt; t++) {
                  C_re[index_4d(0,m,n,t ,Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] = BB_0_re[index_3d(m,n,t,Nsnk,Nt)];
                  C_im[index_4d(0,m,n,t ,Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] = BB_0_im[index_3d(m,n,t,Nsnk,Nt)];
                  C_re[index_4d(1,m,n,t ,Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] = BB_r1_re[index_3d(m,n,t,Nsnk,Nt)];
                  C_im[index_4d(1,m,n,t ,Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] = BB_r1_im[index_3d(m,n,t,Nsnk,Nt)];
                  C_re[index_4d(2,m,n,t ,Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] = BB_r2_re[index_3d(m,n,t,Nsnk,Nt)];
                  C_im[index_4d(2,m,n,t ,Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] = BB_r2_im[index_3d(m,n,t,Nsnk,Nt)];
                  C_re[index_4d(3,m,n,t ,Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] = BB_r3_re[index_3d(m,n,t,Nsnk,Nt)];
                  C_im[index_4d(3,m,n,t ,Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] = BB_r3_im[index_3d(m,n,t,Nsnk,Nt)];
               }
            }
         }
         double end_time = rtclock();
	 total_time_common += (end_time - start_time);

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
         free(BB_0_re);
         free(BB_0_im);
         free(BB_r1_re);
         free(BB_r1_im);
         free(BB_r2_re);
         free(BB_r2_im);
         free(BB_r3_re);
         free(BB_r3_im);
         printf("made BB-BB \n");
      }
      if (NsnkHex > 0) {
         // BB_H 
         double* BB_H_0_re = malloc(Nsrc * NsnkHex * Nt * sizeof (double));
         double* BB_H_0_im = malloc(Nsrc * NsnkHex * Nt * sizeof (double));
         double* BB_H_r1_re = malloc(Nsrc * NsnkHex * Nt * sizeof (double));
         double* BB_H_r1_im = malloc(Nsrc * NsnkHex * Nt * sizeof (double));
         double* BB_H_r2_re = malloc(Nsrc * NsnkHex * Nt * sizeof (double));
         double* BB_H_r2_im = malloc(Nsrc * NsnkHex * Nt * sizeof (double));
         double* BB_H_r3_re = malloc(Nsrc * NsnkHex * Nt * sizeof (double));
         double* BB_H_r3_im = malloc(Nsrc * NsnkHex * Nt * sizeof (double));

	 double start_time = rtclock();
         for (m=0; m<Nsrc; m++) {
            for (n=0; n<NsnkHex; n++) {
               for (t=0; t<Nt; t++) {
                  BB_H_0_re[index_3d(m,n,t ,NsnkHex,Nt)] = 0.0;
                  BB_H_0_im[index_3d(m,n,t ,NsnkHex,Nt)] = 0.0;
                  BB_H_r1_re[index_3d(m,n,t ,NsnkHex,Nt)] = 0.0;
                  BB_H_r1_im[index_3d(m,n,t ,NsnkHex,Nt)] = 0.0;
                  BB_H_r2_re[index_3d(m,n,t ,NsnkHex,Nt)] = 0.0;
                  BB_H_r2_im[index_3d(m,n,t ,NsnkHex,Nt)] = 0.0;
                  BB_H_r3_re[index_3d(m,n,t ,NsnkHex,Nt)] = 0.0;
                  BB_H_r3_im[index_3d(m,n,t ,NsnkHex,Nt)] = 0.0;
               }
            }
         }
         make_dibaryon_hex_correlator(BB_H_0_re, BB_H_0_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B2_Blocal_r2_re, B2_Blocal_r2_im, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_1, snk_spin_weights_1, snk_weights_1, hex_snk_psi_re, hex_snk_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,Nsrc,NsnkHex,Nperms);
         make_dibaryon_hex_correlator(BB_H_0_re, BB_H_0_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B2_Blocal_r2_re, B2_Blocal_r2_im, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_2, snk_spin_weights_2, snk_weights_2, hex_snk_psi_re, hex_snk_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,Nsrc,NsnkHex,Nperms);
         make_dibaryon_hex_correlator(BB_H_0_re, BB_H_0_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B2_Blocal_r1_re, B2_Blocal_r1_im, perms, sigs, -1.0*overall_weight/sqrt(2.0), snk_color_weights_1, snk_spin_weights_1, snk_weights_1, hex_snk_psi_re, hex_snk_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,Nsrc,NsnkHex,Nperms);
         make_dibaryon_hex_correlator(BB_H_0_re, BB_H_0_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B2_Blocal_r1_re, B2_Blocal_r1_im, perms, sigs, -1.0*overall_weight/sqrt(2.0), snk_color_weights_2, snk_spin_weights_2, snk_weights_2, hex_snk_psi_re, hex_snk_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,Nsrc,NsnkHex,Nperms);
         make_dibaryon_hex_correlator(BB_H_r1_re, BB_H_r1_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B2_Blocal_r1_re, B2_Blocal_r1_im, perms, sigs, overall_weight, snk_color_weights_r1, snk_spin_weights_r1, snk_weights_r1, hex_snk_psi_re, hex_snk_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,Nsrc,NsnkHex,Nperms);
         make_dibaryon_hex_correlator(BB_H_r2_re, BB_H_r2_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B2_Blocal_r2_re, B2_Blocal_r2_im, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_r2_1, snk_spin_weights_r2_1, snk_weights_r2_1, hex_snk_psi_re, hex_snk_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,Nsrc,NsnkHex,Nperms);
         make_dibaryon_hex_correlator(BB_H_r2_re, BB_H_r2_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B2_Blocal_r2_re, B2_Blocal_r2_im, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_r2_2, snk_spin_weights_r2_2, snk_weights_r2_2, hex_snk_psi_re, hex_snk_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,Nsrc,NsnkHex,Nperms);
         make_dibaryon_hex_correlator(BB_H_r2_re, BB_H_r2_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B2_Blocal_r1_re, B2_Blocal_r1_im, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_r2_1, snk_spin_weights_r2_1, snk_weights_r2_1, hex_snk_psi_re, hex_snk_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,Nsrc,NsnkHex,Nperms);
         make_dibaryon_hex_correlator(BB_H_r2_re, BB_H_r2_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B2_Blocal_r1_re, B2_Blocal_r1_im, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_r2_2, snk_spin_weights_r2_2, snk_weights_r2_2, hex_snk_psi_re, hex_snk_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,Nsrc,NsnkHex,Nperms);
         make_dibaryon_hex_correlator(BB_H_r3_re, BB_H_r3_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B2_Blocal_r2_re, B2_Blocal_r2_im, perms, sigs, overall_weight, snk_color_weights_r3, snk_spin_weights_r3, snk_weights_r3, hex_snk_psi_re, hex_snk_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,Nsrc,NsnkHex,Nperms);
         if (space_symmetric != 0) {
            make_dibaryon_hex_correlator(BB_H_0_re, BB_H_0_im, B2_Blocal_r1_re, B2_Blocal_r1_im, B1_Blocal_r2_re, B1_Blocal_r2_im, perms, sigs, space_symmetric*overall_weight/sqrt(2.0), snk_color_weights_1, snk_spin_weights_1, snk_weights_1, hex_snk_psi_re, hex_snk_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,Nsrc,NsnkHex,Nperms);
            make_dibaryon_hex_correlator(BB_H_0_re, BB_H_0_im, B2_Blocal_r1_re, B2_Blocal_r1_im, B1_Blocal_r2_re, B1_Blocal_r2_im, perms, sigs, space_symmetric*overall_weight/sqrt(2.0), snk_color_weights_2, snk_spin_weights_2, snk_weights_2, hex_snk_psi_re, hex_snk_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,Nsrc,NsnkHex,Nperms);
            make_dibaryon_hex_correlator(BB_H_0_re, BB_H_0_im, B2_Blocal_r2_re, B2_Blocal_r2_im, B1_Blocal_r1_re, B1_Blocal_r1_im, perms, sigs, -1.0*space_symmetric*overall_weight/sqrt(2.0), snk_color_weights_1, snk_spin_weights_1, snk_weights_1, hex_snk_psi_re, hex_snk_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,Nsrc,NsnkHex,Nperms);
            make_dibaryon_hex_correlator(BB_H_0_re, BB_H_0_im, B2_Blocal_r2_re, B2_Blocal_r2_im, B1_Blocal_r1_re, B1_Blocal_r1_im, perms, sigs, -1.0*space_symmetric*overall_weight/sqrt(2.0), snk_color_weights_2, snk_spin_weights_2, snk_weights_2, hex_snk_psi_re, hex_snk_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,Nsrc,NsnkHex,Nperms);
            make_dibaryon_hex_correlator(BB_H_r1_re, BB_H_r1_im, B2_Blocal_r1_re, B2_Blocal_r1_im, B1_Blocal_r1_re, B1_Blocal_r1_im, perms, sigs, space_symmetric*overall_weight, snk_color_weights_r1, snk_spin_weights_r1, snk_weights_r1, hex_snk_psi_re, hex_snk_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,Nsrc,NsnkHex,Nperms);
            make_dibaryon_hex_correlator(BB_H_r2_re, BB_H_r2_im, B2_Blocal_r1_re, B2_Blocal_r1_im, B1_Blocal_r2_re, B1_Blocal_r2_im, perms, sigs, space_symmetric*overall_weight/sqrt(2.0), snk_color_weights_r2_1, snk_spin_weights_r2_1, snk_weights_r2_1, hex_snk_psi_re, hex_snk_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,Nsrc,NsnkHex,Nperms);
            make_dibaryon_hex_correlator(BB_H_r2_re, BB_H_r2_im, B2_Blocal_r1_re, B2_Blocal_r1_im, B1_Blocal_r2_re, B1_Blocal_r2_im, perms, sigs, space_symmetric*overall_weight/sqrt(2.0), snk_color_weights_r2_2, snk_spin_weights_r2_2, snk_weights_r2_2, hex_snk_psi_re, hex_snk_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,Nsrc,NsnkHex,Nperms);
            make_dibaryon_hex_correlator(BB_H_r2_re, BB_H_r2_im, B2_Blocal_r2_re, B2_Blocal_r2_im, B1_Blocal_r1_re, B1_Blocal_r1_im, perms, sigs, space_symmetric*overall_weight/sqrt(2.0), snk_color_weights_r2_1, snk_spin_weights_r2_1, snk_weights_r2_1, hex_snk_psi_re, hex_snk_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,Nsrc,NsnkHex,Nperms);
            make_dibaryon_hex_correlator(BB_H_r2_re, BB_H_r2_im, B2_Blocal_r2_re, B2_Blocal_r2_im, B1_Blocal_r1_re, B1_Blocal_r1_im, perms, sigs, space_symmetric*overall_weight/sqrt(2.0), snk_color_weights_r2_2, snk_spin_weights_r2_2, snk_weights_r2_2, hex_snk_psi_re, hex_snk_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,Nsrc,NsnkHex,Nperms);
            make_dibaryon_hex_correlator(BB_H_r3_re, BB_H_r3_im, B2_Blocal_r2_re, B2_Blocal_r2_im, B1_Blocal_r2_re, B1_Blocal_r2_im, perms, sigs, space_symmetric*overall_weight, snk_color_weights_r3, snk_spin_weights_r3, snk_weights_r3, hex_snk_psi_re, hex_snk_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,Nsrc,NsnkHex,Nperms);
         }
         for (m=0; m<Nsrc; m++) {
            for (n=0; n<NsnkHex; n++) {
               for (t=0; t<Nt; t++) {
                  C_re[index_4d(0,m,Nsnk+n,t ,Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] = BB_H_0_re[index_3d(m,n,t ,Nsnk,Nt)];
                  C_im[index_4d(0,m,Nsnk+n,t ,Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] = BB_H_0_im[index_3d(m,n,t ,Nsnk,Nt)];
                  C_re[index_4d(1,m,Nsnk+n,t ,Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] = BB_H_r1_re[index_3d(m,n,t ,Nsnk,Nt)];
                  C_im[index_4d(1,m,Nsnk+n,t ,Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] = BB_H_r1_im[index_3d(m,n,t ,Nsnk,Nt)];
                  C_re[index_4d(2,m,Nsnk+n,t ,Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] = BB_H_r2_re[index_3d(m,n,t ,Nsnk,Nt)];
                  C_im[index_4d(2,m,Nsnk+n,t ,Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] = BB_H_r2_im[index_3d(m,n,t ,Nsnk,Nt)];
                  C_re[index_4d(3,m,Nsnk+n,t ,Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] = BB_H_r3_re[index_3d(m,n,t ,Nsnk,Nt)];
                  C_im[index_4d(3,m,Nsnk+n,t ,Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] = BB_H_r3_im[index_3d(m,n,t ,Nsnk,Nt)];
               }
            }
         } 
         if (snk_entangled != 0 && Nsrc == Nsnk && NsrcHex == NsnkHex) {
            for (m=0; m<NsrcHex; m++) {
               for (n=0; n<Nsnk; n++) {
                  for (t=0; t<Nt; t++) {
                     C_re[index_4d(0,Nsrc+m,n,t ,Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] = BB_H_0_re[index_3d(m,n,t,Nsnk,Nt)];
                     C_im[index_4d(0,Nsrc+m,n,t ,Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] = -BB_H_0_im[index_3d(m,n,t,Nsnk,Nt)];
                     C_re[index_4d(1,Nsrc+m,n,t ,Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] = BB_H_r1_re[index_3d(m,n,t,Nsnk,Nt)];
                     C_im[index_4d(1,Nsrc+m,n,t ,Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] = -BB_H_r1_im[index_3d(m,n,t,Nsnk,Nt)];
                     C_re[index_4d(2,Nsrc+m,n,t ,Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] = BB_H_r2_re[index_3d(m,n,t,Nsnk,Nt)];
                     C_im[index_4d(2,Nsrc+m,n,t ,Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] = -BB_H_r2_im[index_3d(m,n,t,Nsnk,Nt)];
                     C_re[index_4d(3,Nsrc+m,n,t ,Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] = BB_H_r3_re[index_3d(m,n,t,Nsnk,Nt)];
                     C_im[index_4d(3,Nsrc+m,n,t ,Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] = -BB_H_r3_im[index_3d(m,n,t,Nsnk,Nt)];
                  }
               }
            }
         } 
         double end_time = rtclock();
	 total_time_common += (end_time - start_time);

         free(BB_H_0_re);
         free(BB_H_0_im);
         free(BB_H_r1_re);
         free(BB_H_r1_im);
         free(BB_H_r2_re);
         free(BB_H_r2_im);
         free(BB_H_r3_re);
         free(BB_H_r3_im);
         printf("made BB-H \n");
      }
      free(B1_Blocal_r1_re);
      free(B1_Blocal_r1_im);
      free(B1_Blocal_r2_re);
      free(B1_Blocal_r2_im);
      free(B2_Blocal_r1_re);
      free(B2_Blocal_r1_im);
      free(B2_Blocal_r2_re);
      free(B2_Blocal_r2_im);
   }
   if (NsrcHex > 0) {
      if (NsnkHex > 0) {
         // H_H 
         double* H_0_re = malloc(NsrcHex * NsnkHex * Nt * sizeof (double));
         double* H_0_im = malloc(NsrcHex * NsnkHex * Nt * sizeof (double));
         double* H_r1_re = malloc(NsrcHex * NsnkHex * Nt * sizeof (double));
         double* H_r1_im = malloc(NsrcHex * NsnkHex * Nt * sizeof (double));
         double* H_r2_re = malloc(NsrcHex * NsnkHex * Nt * sizeof (double));
         double* H_r2_im = malloc(NsrcHex * NsnkHex * Nt * sizeof (double));
         double* H_r3_re = malloc(NsrcHex * NsnkHex * Nt * sizeof (double));
         double* H_r3_im = malloc(NsrcHex * NsnkHex * Nt * sizeof (double));

	 double start_time = rtclock();
         for (m=0; m<NsrcHex; m++) {
            for (n=0; n<NsnkHex; n++) {
               for (t=0; t<Nt; t++) {
                  H_0_re[index_3d(m,n,t ,NsnkHex,Nt)] = 0.0;
                  H_0_im[index_3d(m,n,t ,NsnkHex,Nt)] = 0.0;
                  H_r1_re[index_3d(m,n,t ,NsnkHex,Nt)] = 0.0;
                  H_r1_im[index_3d(m,n,t ,NsnkHex,Nt)] = 0.0;
                  H_r2_re[index_3d(m,n,t ,NsnkHex,Nt)] = 0.0;
                  H_r2_im[index_3d(m,n,t ,NsnkHex,Nt)] = 0.0;
                  H_r3_re[index_3d(m,n,t ,NsnkHex,Nt)] = 0.0;
                  H_r3_im[index_3d(m,n,t ,NsnkHex,Nt)] = 0.0;
               }
            }
         }
         make_hex_correlator(H_0_re, H_0_im, B1_prop_re, B1_prop_im, B2_prop_re, B2_prop_im, perms, sigs, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, overall_weight/sqrt(2.0), snk_color_weights_1, snk_spin_weights_1, snk_weights_1, hex_src_psi_re, hex_src_psi_im, hex_snk_psi_re, hex_snk_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,NsrcHex,NsnkHex,Nperms);
         make_hex_correlator(H_0_re, H_0_im, B1_prop_re, B1_prop_im, B2_prop_re, B2_prop_im, perms, sigs, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, -1.0*overall_weight/sqrt(2.0), snk_color_weights_1, snk_spin_weights_1, snk_weights_1, hex_src_psi_re, hex_src_psi_im, hex_snk_psi_re, hex_snk_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,NsrcHex,NsnkHex,Nperms);
         make_hex_correlator(H_0_re, H_0_im, B1_prop_re, B1_prop_im, B2_prop_re, B2_prop_im, perms, sigs, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, overall_weight/sqrt(2.0), snk_color_weights_2, snk_spin_weights_2, snk_weights_2, hex_src_psi_re, hex_src_psi_im, hex_snk_psi_re, hex_snk_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,NsrcHex,NsnkHex,Nperms);
         make_hex_correlator(H_0_re, H_0_im, B1_prop_re, B1_prop_im, B2_prop_re, B2_prop_im, perms, sigs, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, -1.0*overall_weight/sqrt(2.0), snk_color_weights_2, snk_spin_weights_2, snk_weights_2, hex_src_psi_re, hex_src_psi_im, hex_snk_psi_re, hex_snk_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,NsrcHex,NsnkHex,Nperms);
         make_hex_correlator(H_r1_re, H_r1_im, B1_prop_re, B1_prop_im, B2_prop_re, B2_prop_im, perms, sigs, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, overall_weight, snk_color_weights_r1, snk_spin_weights_r1, snk_weights_r1, hex_src_psi_re, hex_src_psi_im, hex_snk_psi_re, hex_snk_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,NsrcHex,NsnkHex,Nperms);
         make_hex_correlator(H_r2_re, H_r2_im, B1_prop_re, B1_prop_im, B2_prop_re, B2_prop_im, perms, sigs, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, overall_weight/sqrt(2.0), snk_color_weights_r2_1, snk_spin_weights_r2_1, snk_weights_r2_1, hex_src_psi_re, hex_src_psi_im, hex_snk_psi_re, hex_snk_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,NsrcHex,NsnkHex,Nperms);
         make_hex_correlator(H_r2_re, H_r2_im, B1_prop_re, B1_prop_im, B2_prop_re, B2_prop_im, perms, sigs, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, overall_weight/sqrt(2.0), snk_color_weights_r2_1, snk_spin_weights_r2_1, snk_weights_r2_1, hex_src_psi_re, hex_src_psi_im, hex_snk_psi_re, hex_snk_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,NsrcHex,NsnkHex,Nperms);
         make_hex_correlator(H_r2_re, H_r2_im, B1_prop_re, B1_prop_im, B2_prop_re, B2_prop_im, perms, sigs, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, overall_weight/sqrt(2.0), snk_color_weights_r2_2, snk_spin_weights_r2_2, snk_weights_r2_2, hex_src_psi_re, hex_src_psi_im, hex_snk_psi_re, hex_snk_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,NsrcHex,NsnkHex,Nperms);
         make_hex_correlator(H_r2_re, H_r2_im, B1_prop_re, B1_prop_im, B2_prop_re, B2_prop_im, perms, sigs, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, overall_weight/sqrt(2.0), snk_color_weights_r2_2, snk_spin_weights_r2_2, snk_weights_r2_2, hex_src_psi_re, hex_src_psi_im, hex_snk_psi_re, hex_snk_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,NsrcHex,NsnkHex,Nperms);
         make_hex_correlator(H_r3_re, H_r3_im, B1_prop_re, B1_prop_im, B2_prop_re, B2_prop_im, perms, sigs, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, overall_weight, snk_color_weights_r3, snk_spin_weights_r3, snk_weights_r3, hex_src_psi_re, hex_src_psi_im, hex_snk_psi_re, hex_snk_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,NsrcHex,NsnkHex,Nperms);
         for (m=0; m<NsrcHex; m++) {
            for (n=0; n<NsnkHex; n++) {
               for (t=0; t<Nt; t++) {
                  C_re[index_4d(0,Nsrc+m,Nsnk+n,t ,Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] = H_0_re[index_3d(m,n,t ,Nsnk,Nt)];
                  C_im[index_4d(0,Nsrc+m,Nsnk+n,t ,Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] = H_0_im[index_3d(m,n,t ,Nsnk,Nt)];
                  C_re[index_4d(1,Nsrc+m,Nsnk+n,t ,Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] = H_r1_re[index_3d(m,n,t ,Nsnk,Nt)];
                  C_im[index_4d(1,Nsrc+m,Nsnk+n,t ,Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] = H_r1_im[index_3d(m,n,t ,Nsnk,Nt)];
                  C_re[index_4d(2,Nsrc+m,Nsnk+n,t ,Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] = H_r2_re[index_3d(m,n,t ,Nsnk,Nt)];
                  C_im[index_4d(2,Nsrc+m,Nsnk+n,t ,Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] = H_r2_im[index_3d(m,n,t ,Nsnk,Nt)];
                  C_re[index_4d(3,Nsrc+m,Nsnk+n,t ,Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] = H_r3_re[index_3d(m,n,t ,Nsnk,Nt)];
                  C_im[index_4d(3,Nsrc+m,Nsnk+n,t ,Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] = H_r3_im[index_3d(m,n,t ,Nsnk,Nt)];
               }
            }
         }
         double end_time = rtclock();
	 total_time_common += (end_time - start_time);

         free(H_0_re);
         free(H_0_im);
         free(H_r1_re);
         free(H_r1_im);
         free(H_r2_re);
         free(H_r2_im);
         free(H_r3_re);
         free(H_r3_im);
         printf("made H-H \n");
      }
      if (Nsnk > 0) {
         if (snk_entangled == 0) {
            // H_BB 
            double* snk_B1_Blocal_r1_re = malloc(Nt * Nc * Ns * Nc * Ns * Vsrc * Nc * Ns * Nsnk * sizeof (double));
            double* snk_B1_Blocal_r1_im = malloc(Nt * Nc * Ns * Nc * Ns * Vsrc * Nc * Ns * Nsnk * sizeof (double));
            double* snk_B1_Blocal_r2_re = malloc(Nt * Nc * Ns * Nc * Ns * Vsrc * Nc * Ns * Nsnk * sizeof (double));
            double* snk_B1_Blocal_r2_im = malloc(Nt * Nc * Ns * Nc * Ns * Vsrc * Nc * Ns * Nsnk * sizeof (double));
            double* snk_B2_Blocal_r1_re = malloc(Nt * Nc * Ns * Nc * Ns * Vsrc * Nc * Ns * Nsnk * sizeof (double));
            double* snk_B2_Blocal_r1_im = malloc(Nt * Nc * Ns * Nc * Ns * Vsrc * Nc * Ns * Nsnk * sizeof (double));
            double* snk_B2_Blocal_r2_re = malloc(Nt * Nc * Ns * Nc * Ns * Vsrc * Nc * Ns * Nsnk * sizeof (double));
            double* snk_B2_Blocal_r2_im = malloc(Nt * Nc * Ns * Nc * Ns * Vsrc * Nc * Ns * Nsnk * sizeof (double));

	    double* H_BB_0_re = malloc(NsrcHex * Nsnk * Nt * sizeof (double));
            double* H_BB_0_im = malloc(NsrcHex * Nsnk * Nt * sizeof (double));
            double* H_BB_r1_re = malloc(NsrcHex * Nsnk * Nt * sizeof (double));
            double* H_BB_r1_im = malloc(NsrcHex * Nsnk * Nt * sizeof (double));
            double* H_BB_r2_re = malloc(NsrcHex * Nsnk * Nt * sizeof (double));
            double* H_BB_r2_im = malloc(NsrcHex * Nsnk * Nt * sizeof (double));
            double* H_BB_r3_re = malloc(NsrcHex * Nsnk * Nt * sizeof (double));
            double* H_BB_r3_im = malloc(NsrcHex * Nsnk * Nt * sizeof (double));

	    double start_time = rtclock();
            make_local_snk_block(snk_B1_Blocal_r1_re, snk_B1_Blocal_r1_im, B1_prop_re, B1_prop_im, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, snk_psi_B1_re, snk_psi_B1_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,Nsnk);
            make_local_snk_block(snk_B1_Blocal_r2_re, snk_B1_Blocal_r2_im, B1_prop_re, B1_prop_im, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, snk_psi_B1_re, snk_psi_B1_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,Nsnk);
            make_local_snk_block(snk_B2_Blocal_r1_re, snk_B2_Blocal_r1_im, B2_prop_re, B2_prop_im, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, snk_psi_B2_re, snk_psi_B2_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,Nsnk);
            make_local_snk_block(snk_B2_Blocal_r2_re, snk_B2_Blocal_r2_im, B2_prop_re, B2_prop_im, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, snk_psi_B2_re, snk_psi_B2_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,Nsnk);
            for (m=0; m<NsrcHex; m++) {
               for (n=0; n<Nsnk; n++) {
                  for (t=0; t<Nt; t++) {
                     H_BB_0_re[index_3d(m,n,t,Nsnk,Nt)] = 0.0;
                     H_BB_0_im[index_3d(m,n,t,Nsnk,Nt)] = 0.0;
                     H_BB_r1_re[index_3d(m,n,t,Nsnk,Nt)] = 0.0;
                     H_BB_r1_im[index_3d(m,n,t,Nsnk,Nt)] = 0.0;
                     H_BB_r2_re[index_3d(m,n,t,Nsnk,Nt)] = 0.0;
                     H_BB_r2_im[index_3d(m,n,t,Nsnk,Nt)] = 0.0;
                     H_BB_r3_re[index_3d(m,n,t,Nsnk,Nt)] = 0.0;
                     H_BB_r3_im[index_3d(m,n,t,Nsnk,Nt)] = 0.0; 
                  }
               }
            }
            make_hex_dibaryon_correlator(H_BB_0_re, H_BB_0_im, snk_B1_Blocal_r1_re, snk_B1_Blocal_r1_im, snk_B2_Blocal_r2_re, snk_B2_Blocal_r2_im, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_1, snk_spin_weights_1, snk_weights_1, hex_src_psi_re, hex_src_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,NsrcHex,Nsnk,Nperms);
            make_hex_dibaryon_correlator(H_BB_0_re, H_BB_0_im, snk_B1_Blocal_r1_re, snk_B1_Blocal_r1_im, snk_B2_Blocal_r2_re, snk_B2_Blocal_r2_im, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_2, snk_spin_weights_2, snk_weights_2, hex_src_psi_re, hex_src_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,NsrcHex,Nsnk,Nperms);
            make_hex_dibaryon_correlator(H_BB_0_re, H_BB_0_im, snk_B1_Blocal_r2_re, snk_B1_Blocal_r2_im, snk_B2_Blocal_r1_re, snk_B2_Blocal_r1_im, perms, sigs, -1.0*overall_weight/sqrt(2.0), snk_color_weights_1, snk_spin_weights_1, snk_weights_1, hex_src_psi_re, hex_src_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,NsrcHex,Nsnk,Nperms);
            make_hex_dibaryon_correlator(H_BB_0_re, H_BB_0_im, snk_B1_Blocal_r2_re, snk_B1_Blocal_r2_im, snk_B2_Blocal_r1_re, snk_B2_Blocal_r1_im, perms, sigs, -1.0*overall_weight/sqrt(2.0), snk_color_weights_2, snk_spin_weights_2, snk_weights_2, hex_src_psi_re, hex_src_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,NsrcHex,Nsnk,Nperms);
            make_hex_dibaryon_correlator(H_BB_r1_re, H_BB_r1_im, snk_B1_Blocal_r1_re, snk_B1_Blocal_r1_im, snk_B2_Blocal_r1_re, snk_B2_Blocal_r1_im, perms, sigs, overall_weight, snk_color_weights_r1, snk_spin_weights_r1, snk_weights_r1, hex_src_psi_re, hex_src_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,NsrcHex,Nsnk,Nperms);
            make_hex_dibaryon_correlator(H_BB_r2_re, H_BB_r2_im, snk_B1_Blocal_r1_re, snk_B1_Blocal_r1_im, snk_B2_Blocal_r2_re, snk_B2_Blocal_r2_im, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_r2_1, snk_spin_weights_r2_1, snk_weights_r2_1, hex_src_psi_re, hex_src_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,NsrcHex,Nsnk,Nperms);
            make_hex_dibaryon_correlator(H_BB_r2_re, H_BB_r2_im, snk_B1_Blocal_r1_re, snk_B1_Blocal_r1_im, snk_B2_Blocal_r2_re, snk_B2_Blocal_r2_im, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_r2_2, snk_spin_weights_r2_2, snk_weights_r2_2, hex_src_psi_re, hex_src_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,NsrcHex,Nsnk,Nperms);
            make_hex_dibaryon_correlator(H_BB_r2_re, H_BB_r2_im, snk_B1_Blocal_r2_re, snk_B1_Blocal_r2_im, snk_B2_Blocal_r1_re, snk_B2_Blocal_r1_im, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_r2_1, snk_spin_weights_r2_1, snk_weights_r2_1, hex_src_psi_re, hex_src_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,NsrcHex,Nsnk,Nperms);
            make_hex_dibaryon_correlator(H_BB_r2_re, H_BB_r2_im, snk_B1_Blocal_r2_re, snk_B1_Blocal_r2_im, snk_B2_Blocal_r1_re, snk_B2_Blocal_r1_im, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_r2_2, snk_spin_weights_r2_2, snk_weights_r2_2, hex_src_psi_re, hex_src_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,NsrcHex,Nsnk,Nperms);
            make_hex_dibaryon_correlator(H_BB_r3_re, H_BB_r3_im, snk_B1_Blocal_r2_re, snk_B1_Blocal_r2_im, snk_B2_Blocal_r2_re, snk_B2_Blocal_r2_im, perms, sigs, overall_weight, snk_color_weights_r3, snk_spin_weights_r3, snk_weights_r3, hex_src_psi_re, hex_src_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,NsrcHex,Nsnk,Nperms);
            if (space_symmetric != 0) {
               make_hex_dibaryon_correlator(H_BB_0_re, H_BB_0_im, snk_B2_Blocal_r1_re, snk_B2_Blocal_r1_im, snk_B1_Blocal_r2_re, snk_B1_Blocal_r2_im, perms, sigs, space_symmetric*overall_weight/sqrt(2.0), snk_color_weights_1, snk_spin_weights_1, snk_weights_1, hex_src_psi_re, hex_src_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,NsrcHex,Nsnk,Nperms);
               make_hex_dibaryon_correlator(H_BB_0_re, H_BB_0_im, snk_B2_Blocal_r1_re, snk_B2_Blocal_r1_im, snk_B1_Blocal_r2_re, snk_B1_Blocal_r2_im, perms, sigs, space_symmetric*overall_weight/sqrt(2.0), snk_color_weights_2, snk_spin_weights_2, snk_weights_2, hex_src_psi_re, hex_src_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,NsrcHex,Nsnk,Nperms);
               make_hex_dibaryon_correlator(H_BB_0_re, H_BB_0_im, snk_B2_Blocal_r2_re, snk_B2_Blocal_r2_im, snk_B1_Blocal_r1_re, snk_B1_Blocal_r1_im, perms, sigs, -1.0*space_symmetric*overall_weight/sqrt(2.0), snk_color_weights_1, snk_spin_weights_1, snk_weights_1, hex_src_psi_re, hex_src_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,NsrcHex,Nsnk,Nperms);
               make_hex_dibaryon_correlator(H_BB_0_re, H_BB_0_im, snk_B2_Blocal_r2_re, snk_B2_Blocal_r2_im, snk_B1_Blocal_r1_re, snk_B1_Blocal_r1_im, perms, sigs, -1.0*space_symmetric*overall_weight/sqrt(2.0), snk_color_weights_2, snk_spin_weights_2, snk_weights_2, hex_src_psi_re, hex_src_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,NsrcHex,Nsnk,Nperms);
               make_hex_dibaryon_correlator(H_BB_r1_re, H_BB_r1_im, snk_B2_Blocal_r1_re, snk_B2_Blocal_r1_im, snk_B1_Blocal_r1_re, snk_B1_Blocal_r1_im, perms, sigs, space_symmetric*overall_weight, snk_color_weights_r1, snk_spin_weights_r1, snk_weights_r1, hex_src_psi_re, hex_src_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,NsrcHex,Nsnk,Nperms);
               make_hex_dibaryon_correlator(H_BB_r2_re, H_BB_r2_im, snk_B2_Blocal_r1_re, snk_B2_Blocal_r1_im, snk_B1_Blocal_r2_re, snk_B1_Blocal_r2_im, perms, sigs, space_symmetric*overall_weight/sqrt(2.0), snk_color_weights_r2_1, snk_spin_weights_r2_1, snk_weights_r2_1, hex_src_psi_re, hex_src_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,NsrcHex,Nsnk,Nperms);
               make_hex_dibaryon_correlator(H_BB_r2_re, H_BB_r2_im, snk_B2_Blocal_r1_re, snk_B2_Blocal_r1_im, snk_B1_Blocal_r2_re, snk_B1_Blocal_r2_im, perms, sigs, space_symmetric*overall_weight/sqrt(2.0), snk_color_weights_r2_2, snk_spin_weights_r2_2, snk_weights_r2_2, hex_src_psi_re, hex_src_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,NsrcHex,Nsnk,Nperms);
               make_hex_dibaryon_correlator(H_BB_r2_re, H_BB_r2_im, snk_B2_Blocal_r2_re, snk_B2_Blocal_r2_im, snk_B1_Blocal_r1_re, snk_B1_Blocal_r1_im, perms, sigs, space_symmetric*overall_weight/sqrt(2.0), snk_color_weights_r2_1, snk_spin_weights_r2_1, snk_weights_r2_1, hex_src_psi_re, hex_src_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,NsrcHex,Nsnk,Nperms);
               make_hex_dibaryon_correlator(H_BB_r2_re, H_BB_r2_im, snk_B2_Blocal_r2_re, snk_B2_Blocal_r2_im, snk_B1_Blocal_r1_re, snk_B1_Blocal_r1_im, perms, sigs, space_symmetric*overall_weight/sqrt(2.0), snk_color_weights_r2_2, snk_spin_weights_r2_2, snk_weights_r2_2, hex_src_psi_re, hex_src_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,NsrcHex,Nsnk,Nperms);
               make_hex_dibaryon_correlator(H_BB_r3_re, H_BB_r3_im, snk_B2_Blocal_r2_re, snk_B2_Blocal_r2_im, snk_B1_Blocal_r2_re, snk_B1_Blocal_r2_im, perms, sigs, space_symmetric*overall_weight, snk_color_weights_r3, snk_spin_weights_r3, snk_weights_r3, hex_src_psi_re, hex_src_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,NsrcHex,Nsnk,Nperms);
            }
            for (m=0; m<NsrcHex; m++) {
               for (n=0; n<Nsnk; n++) {
                  for (t=0; t<Nt; t++) {
                     C_re[index_4d(0,Nsrc+m,n,t ,Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] = H_BB_0_re[index_3d(m,n,t,Nsnk,Nt)];
                     C_im[index_4d(0,Nsrc+m,n,t ,Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] = H_BB_0_im[index_3d(m,n,t,Nsnk,Nt)];
                     C_re[index_4d(1,Nsrc+m,n,t ,Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] = H_BB_r1_re[index_3d(m,n,t,Nsnk,Nt)];
                     C_im[index_4d(1,Nsrc+m,n,t ,Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] = H_BB_r1_im[index_3d(m,n,t,Nsnk,Nt)];
                     C_re[index_4d(2,Nsrc+m,n,t ,Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] = H_BB_r2_re[index_3d(m,n,t,Nsnk,Nt)];
                     C_im[index_4d(2,Nsrc+m,n,t ,Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] = H_BB_r2_im[index_3d(m,n,t,Nsnk,Nt)];
                     C_re[index_4d(3,Nsrc+m,n,t ,Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] = H_BB_r3_re[index_3d(m,n,t,Nsnk,Nt)];
                     C_im[index_4d(3,Nsrc+m,n,t ,Nsrc+NsrcHex,Nsnk+NsnkHex,Nt)] = H_BB_r3_im[index_3d(m,n,t,Nsnk,Nt)];
                  }
               }
            }
	    double end_time = rtclock();
	    total_time_common += (end_time - start_time);

            free(snk_B1_Blocal_r1_re);
            free(snk_B1_Blocal_r1_im);
            free(snk_B1_Blocal_r2_re);
            free(snk_B1_Blocal_r2_im);
            free(snk_B2_Blocal_r1_re);
            free(snk_B2_Blocal_r1_im);
            free(snk_B2_Blocal_r2_re);
            free(snk_B2_Blocal_r2_im);
            free(H_BB_0_re);
            free(H_BB_0_im);
            free(H_BB_r1_re);
            free(H_BB_r1_im);
            free(H_BB_r2_re);
            free(H_BB_r2_im);
            free(H_BB_r3_re);
            free(H_BB_r3_im);
            printf("made H-BB \n");
         }
      }
   }

   if (USE_REFERENCE)
   {
	printf("Total execution time for reference code: %lf\n", (total_time_common + total_time_reference)*1000);
	printf("(optimized part alone: %lf)\n", (total_time_reference)*1000);
   }
   if (USE_TIRAMISU)
   {
	printf("Total execution time for Tiramisu code: %lf\n",  (total_time_common + total_time_tiramisu)*1000);
	printf("(optimized part alone: %lf)\n", (total_time_tiramisu)*1000);
   }
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
    const int Nc,
    const int Ns,
    const int Vsrc,
    const int Vsnk,
    const int Nt,
    const int Nw,
    const int Nq,
    const int Nsrc,
    const int Nsnk) {
   /* indices */
   int n, m, t, x, iC, iS, jC, jS, kC, kS, wnum;
   /* create block */
   double* Blocal_r1_re = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Nsrc * sizeof (double));
   double* Blocal_r1_im = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Nsrc * sizeof (double));
   double* Blocal_r2_re = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Nsrc * sizeof (double));
   double* Blocal_r2_im = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Nsrc * sizeof (double));
   make_local_block(Blocal_r1_re, Blocal_r1_im, prop_re, prop_im, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, src_psi_re, src_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,Nsrc);
   make_local_block(Blocal_r2_re, Blocal_r2_im, prop_re, prop_im, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, src_psi_re, src_psi_im, Nc,Ns,Vsrc,Vsnk,Nt,Nw,Nq,Nsrc);
   /* create baryon correlator */

   for (wnum=0; wnum<Nw; wnum++) {
      iC = src_color_weights_r1[index_2d(wnum,0, Nq)];
      iS = src_spin_weights_r1[index_2d(wnum,0, Nq)];
      jC = src_color_weights_r1[index_2d(wnum,1, Nq)];
      jS = src_spin_weights_r1[index_2d(wnum,1, Nq)];
      kC = src_color_weights_r1[index_2d(wnum,2, Nq)];
      kS = src_spin_weights_r1[index_2d(wnum,2, Nq)];
      for (m=0; m<Nsrc; m++) {
         for (n=0; n<Nsnk; n++) {
            for (t=0; t<Nt; t++) {
               for (x=0; x<Vsnk; x++) {
                  C_re[index_4d(0,m,n,t ,Nsrc,Nsnk,Nt)] += src_weights_r1[wnum] * (snk_psi_re[index_2d(x,n ,Nsnk)] * Blocal_r1_re[Blocal_index(t,iC,iS,kC,kS,x,jC,jS,m ,Nc,Ns,Vsnk,Nsrc)] - snk_psi_im[index_2d(x,n ,Nsnk)] * Blocal_r1_im[Blocal_index(t,iC,iS,kC,kS,x,jC,jS,m ,Nc,Ns,Vsnk,Nsrc)]);
                  C_im[index_4d(0,m,n,t ,Nsrc,Nsnk,Nt)] += src_weights_r1[wnum] * (snk_psi_re[index_2d(x,n ,Nsnk)] * Blocal_r1_im[Blocal_index(t,iC,iS,kC,kS,x,jC,jS,m ,Nc,Ns,Vsnk,Nsrc)] - snk_psi_im[index_2d(x,n ,Nsnk)] * Blocal_r1_re[Blocal_index(t,iC,iS,kC,kS,x,jC,jS,m ,Nc,Ns,Vsnk,Nsrc)]);
               }
            }
         }
      }
   }
   for (wnum=0; wnum<Nw; wnum++) {
      iC = src_color_weights_r2[index_2d(wnum,0, Nq)];
      iS = src_spin_weights_r2[index_2d(wnum,0, Nq)];
      jC = src_color_weights_r2[index_2d(wnum,1, Nq)];
      jS = src_spin_weights_r2[index_2d(wnum,1, Nq)];
      kC = src_color_weights_r2[index_2d(wnum,2, Nq)];
      kS = src_spin_weights_r2[index_2d(wnum,2, Nq)];
      for (m=0; m<Nsrc; m++) {
         for (n=0; n<Nsnk; n++) {
            for (t=0; t<Nt; t++) {
               for (x=0; x<Vsnk; x++) {
                  C_re[index_4d(1,m,n,t ,Nsrc,Nsnk,Nt)] += src_weights_r2[wnum] * (snk_psi_re[index_2d(x,n ,Nsnk)] * Blocal_r2_re[Blocal_index(t,iC,iS,kC,kS,x,jC,jS,m ,Nc,Ns,Vsnk,Nsrc)] - snk_psi_im[index_2d(x,n ,Nsnk)] * Blocal_r2_im[Blocal_index(t,iC,iS,kC,kS,x,jC,jS,m ,Nc,Ns,Vsnk,Nsrc)]);
                  C_im[index_4d(1,m,n,t ,Nsrc,Nsnk,Nt)] += src_weights_r2[wnum] * (snk_psi_re[index_2d(x,n ,Nsnk)] * Blocal_r2_im[Blocal_index(t,iC,iS,kC,kS,x,jC,jS,m ,Nc,Ns,Vsnk,Nsrc)] - snk_psi_im[index_2d(x,n ,Nsnk)] * Blocal_r2_re[Blocal_index(t,iC,iS,kC,kS,x,jC,jS,m ,Nc,Ns,Vsnk,Nsrc)]);
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
