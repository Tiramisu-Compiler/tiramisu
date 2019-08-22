#include <stdio.h> 
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <time.h>

#include "qblocks_2pt_scalar.h"                                       /* DEPS */

/* index functions */
int prop_index(int q, int t, int c1, int s1, int c2, int s2, int y, int x) {
   return x +Vsnk*( y +Vsrc*( c2 +Nc*( s2 +Ns*( c1 +Nc*( s1 +Ns*( t +Nt* q ))))));
}
int Q_index(int t, int c1, int s1, int c2, int s2, int x1, int c3, int s3, int y) {
   return y +Vsrc*( s3 +Ns*( c3 +Nc*( x1 +Vsnk*( s2 +Ns*( c2 +Nc*( s1 +Ns*( c1 +Nc*( t ))))))));
}
int Blocal_index(int t, int c1, int s1, int c2, int s2, int x, int c3, int s3, int m) {
   return m +Nsrc*( s3 +Ns*( c3 +Nc*( x +Vsnk*( s2 +Ns*( c2 +Nc*( s1 +Ns*( c1 +Nc*( t ))))))));
}
int snk_Blocal_index(int t, int c1, int s1, int c2, int s2, int y, int c3, int s3, int n) {
   return n +Nsnk*( s3 +Ns*( c3 +Nc*( y +Vsrc*( s2 +Ns*( c2 +Nc*( s1 +Ns*( c1 +Nc*( t ))))))));
}
int Bsingle_index(int t, int c1, int s1, int c2, int s2, int x1, int c3, int s3, int x2, int m) {
   return m +Nsrc*( x2 +Vsnk*(  s3 +Ns*( c3 +Nc*( x1 +Vsnk*( s2 +Ns*( c2 +Nc*( s1 +Ns*( c1 +Nc*( t )))))))));
}
int Bdouble_index(int t, int c1, int s1, int c2, int s2, int x1, int c3, int s3, int x2, int m) {
   return m +Nsrc*( x2 +Vsnk*(  s3 +Ns*( c3 +Nc*( x1 +Vsnk*( s2 +Ns*( c2 +Nc*( s1 +Ns*( c1 +Nc*( t )))))))));
}
int src_weight_index(int nw, int nq) {
   return nq +Nq* nw;
}
int snk_weight_index(int nb, int nw, int nq) {
   return nq +Nq*( nw +(Nw*Nw)*( nb ));
}
int src_psi_index(int y, int m) {
   return m +Nsrc*( y );
}
int snk_one_psi_index(int x, int n) {
   return n +Nsnk*( x );
}
int snk_psi_index(int x1, int x2, int n) {
   return n +Nsnk*( x1 +Vsnk*( x2 ));
}
int hex_src_psi_index(int y, int m) {
   return m +NsrcHex*( y );
}
int hex_snk_psi_index(int x, int n) {
   return n +NsrcHex*( x );
}
int perm_index(int n, int q) {
   return q +(2*Nq)*( n );
}
int one_correlator_index(int m, int n, int t) {
   return t +Nt*( n +Nsnk*( m ));
}
int one_hex_correlator_index(int m, int n, int t) {
   return t +Nt*( n +NsnkHex*( m ));
}
int correlator_index(int r, int m, int n, int t) {
   return t +Nt*( n +(Nsnk+NsnkHex)*( m +(Nsrc+NsrcHex)* ( r )));
}
int B1_correlator_index(int r, int m, int n, int t) {
   return t +Nt*( n +Nsnk*( m +Nsrc* ( r )));
}

void make_local_block(double* Blocal_re, 
    double* Blocal_im, 
    const double* prop_re,
    const double* prop_im, 
    const int* color_weights, 
    const int* spin_weights, 
    const double* weights, 
    const double* psi_re, 
    const double* psi_im) {
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
                              Blocal_re[Blocal_index(t,iCprime,iSprime,kCprime,kSprime,x,jCprime,jSprime,m)] = 0.0;
                              Blocal_im[Blocal_index(t,iCprime,iSprime,kCprime,kSprime,x,jCprime,jSprime,m)] = 0.0;
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
      iC = color_weights[src_weight_index(wnum,0)];
      iS = spin_weights[src_weight_index(wnum,0)];
      jC = color_weights[src_weight_index(wnum,1)];
      jS = spin_weights[src_weight_index(wnum,1)];
      kC = color_weights[src_weight_index(wnum,2)];
      kS = spin_weights[src_weight_index(wnum,2)];
      for (t=0; t<Nt; t++) {
         for (iCprime=0; iCprime<Nc; iCprime++) {
            for (iSprime=0; iSprime<Ns; iSprime++) {
               for (kCprime=0; kCprime<Nc; kCprime++) {
                  for (kSprime=0; kSprime<Ns; kSprime++) {
                     for (x=0; x<Vsnk; x++) {
                        for (y=0; y<Vsrc; y++) {
                           prop_prod_02_re = weights[wnum] * ( (prop_re[prop_index(0,t,iC,iS,iCprime,iSprime,y,x)] * prop_re[prop_index(2,t,kC,kS,kCprime,kSprime,y,x)] - prop_im[prop_index(0,t,iC,iS,iCprime,iSprime,y,x)] * prop_im[prop_index(2,t,kC,kS,kCprime,kSprime,y,x)]) - (prop_re[prop_index(0,t,iC,iS,kCprime,kSprime,y,x)] * prop_re[prop_index(2,t,kC,kS,iCprime,iSprime,y,x)] - prop_im[prop_index(0,t,iC,iS,kCprime,kSprime,y,x)] * prop_im[prop_index(2,t,kC,kS,iCprime,iSprime,y,x)]) );
                           prop_prod_02_im = weights[wnum] * ( (prop_re[prop_index(0,t,iC,iS,iCprime,iSprime,y,x)] * prop_im[prop_index(2,t,kC,kS,kCprime,kSprime,y,x)] + prop_im[prop_index(0,t,iC,iS,iCprime,iSprime,y,x)] * prop_re[prop_index(2,t,kC,kS,kCprime,kSprime,y,x)]) - (prop_re[prop_index(0,t,iC,iS,kCprime,kSprime,y,x)] * prop_im[prop_index(2,t,kC,kS,iCprime,iSprime,y,x)] + prop_im[prop_index(0,t,iC,iS,kCprime,kSprime,y,x)] * prop_re[prop_index(2,t,kC,kS,iCprime,iSprime,y,x)]) );
                           for (jCprime=0; jCprime<Nc; jCprime++) {
                              for (jSprime=0; jSprime<Ns; jSprime++) {
                                 prop_prod_re = prop_prod_02_re * prop_re[prop_index(1,t,jC,jS,jCprime,jSprime,y,x)] - prop_prod_02_im * prop_im[prop_index(1,t,jC,jS,jCprime,jSprime,y,x)];
                                 prop_prod_im = prop_prod_02_re * prop_im[prop_index(1,t,jC,jS,jCprime,jSprime,y,x)] + prop_prod_02_im * prop_re[prop_index(1,t,jC,jS,jCprime,jSprime,y,x)];
                                 for (m=0; m<Nsrc; m++) {
                                    Blocal_re[Blocal_index(t,iCprime,iSprime,kCprime,kSprime,x,jCprime,jSprime,m)] += psi_re[src_psi_index(y,m)] * prop_prod_re - psi_im[src_psi_index(y,m)] * prop_prod_im;
                                    Blocal_im[Blocal_index(t,iCprime,iSprime,kCprime,kSprime,x,jCprime,jSprime,m)] += psi_re[src_psi_index(y,m)] * prop_prod_im + psi_im[src_psi_index(y,m)] * prop_prod_re;
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
    const double* psi_im) {
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
                              Blocal_re[snk_Blocal_index(t,iCprime,iSprime,kCprime,kSprime,y,jCprime,jSprime,n)] = 0.0;
                              Blocal_im[snk_Blocal_index(t,iCprime,iSprime,kCprime,kSprime,y,jCprime,jSprime,n)] = 0.0;
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
      iC = color_weights[src_weight_index(wnum,0)];
      iS = spin_weights[src_weight_index(wnum,0)];
      jC = color_weights[src_weight_index(wnum,1)];
      jS = spin_weights[src_weight_index(wnum,1)];
      kC = color_weights[src_weight_index(wnum,2)];
      kS = spin_weights[src_weight_index(wnum,2)];
      for (t=0; t<Nt; t++) {
         for (iCprime=0; iCprime<Nc; iCprime++) {
            for (iSprime=0; iSprime<Ns; iSprime++) {
               for (kCprime=0; kCprime<Nc; kCprime++) {
                  for (kSprime=0; kSprime<Ns; kSprime++) {
                     for (y=0; y<Vsrc; y++) {
                        for (x=0; x<Vsnk; x++) {
                           prop_prod_02_re = weights[wnum] * ( (prop_re[prop_index(0,t,iC,iS,iCprime,iSprime,y,x)] * prop_re[prop_index(2,t,kC,kS,kCprime,kSprime,y,x)] - prop_im[prop_index(0,t,iC,iS,iCprime,iSprime,y,x)] * prop_im[prop_index(2,t,kC,kS,kCprime,kSprime,y,x)]) - (prop_re[prop_index(0,t,iC,iS,kCprime,kSprime,y,x)] * prop_re[prop_index(2,t,kC,kS,iCprime,iSprime,y,x)] - prop_im[prop_index(0,t,iC,iS,kCprime,kSprime,y,x)] * prop_im[prop_index(2,t,kC,kS,iCprime,iSprime,y,x)]) );
                           prop_prod_02_im = weights[wnum] * ( (prop_re[prop_index(0,t,iC,iS,iCprime,iSprime,y,x)] * prop_im[prop_index(2,t,kC,kS,kCprime,kSprime,y,x)] + prop_im[prop_index(0,t,iC,iS,iCprime,iSprime,y,x)] * prop_re[prop_index(2,t,kC,kS,kCprime,kSprime,y,x)]) - (prop_re[prop_index(0,t,iC,iS,kCprime,kSprime,y,x)] * prop_im[prop_index(2,t,kC,kS,iCprime,iSprime,y,x)] + prop_im[prop_index(0,t,iC,iS,kCprime,kSprime,y,x)] * prop_re[prop_index(2,t,kC,kS,iCprime,iSprime,y,x)]) );
                           for (jCprime=0; jCprime<Nc; jCprime++) {
                              for (jSprime=0; jSprime<Ns; jSprime++) {
                                 prop_prod_re = prop_prod_02_re * prop_re[prop_index(1,t,jC,jS,jCprime,jSprime,y,x)] - prop_prod_02_im * prop_im[prop_index(1,t,jC,jS,jCprime,jSprime,y,x)];
                                 prop_prod_im = prop_prod_02_re * prop_im[prop_index(1,t,jC,jS,jCprime,jSprime,y,x)] + prop_prod_02_im * prop_re[prop_index(1,t,jC,jS,jCprime,jSprime,y,x)];
                                 for (n=0; n<Nsnk; n++) {
                                    Blocal_re[snk_Blocal_index(t,iCprime,iSprime,kCprime,kSprime,y,jCprime,jSprime,n)] += psi_re[snk_one_psi_index(x,n)] * prop_prod_re - psi_im[snk_one_psi_index(x,n)] * prop_prod_im;
                                    Blocal_im[snk_Blocal_index(t,iCprime,iSprime,kCprime,kSprime,y,jCprime,jSprime,n)] += psi_re[snk_one_psi_index(x,n)] * prop_prod_im + psi_im[snk_one_psi_index(x,n)] * prop_prod_re;
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
    const double* psi_im) {
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
                                 Bsingle_re[Bsingle_index(t,iCprime,iSprime,kCprime,kSprime,x1,jCprime,jSprime,x2,m)] = 0.0;
                                 Bsingle_im[Bsingle_index(t,iCprime,iSprime,kCprime,kSprime,x1,jCprime,jSprime,x2,m)] = 0.0;
                              }
                           }
                        }
                     }
                     for (jC=0; jC<Nc; jC++) {
                        for (jS=0; jS<Ns; jS++) {
                           for (y=0; y<Vsrc; y++) {
                              Q02_re[Q_index(t,iCprime,iSprime,kCprime,kSprime,x1,jC,jS,y)] = 0.0;
                              Q02_im[Q_index(t,iCprime,iSprime,kCprime,kSprime,x1,jC,jS,y)] = 0.0;
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
      iC = color_weights[src_weight_index(wnum,0)];
      iS = spin_weights[src_weight_index(wnum,0)];
      jC = color_weights[src_weight_index(wnum,1)];
      jS = spin_weights[src_weight_index(wnum,1)];
      kC = color_weights[src_weight_index(wnum,2)];
      kS = spin_weights[src_weight_index(wnum,2)];
      for (t=0; t<Nt; t++) {
         for (iCprime=0; iCprime<Nc; iCprime++) {
            for (iSprime=0; iSprime<Ns; iSprime++) {
               for (kCprime=0; kCprime<Nc; kCprime++) {
                  for (kSprime=0; kSprime<Ns; kSprime++) {
                     for (y=0; y<Vsrc; y++) {
                        for (x=0; x<Vsnk; x++) {
                           Q02_re[Q_index(t,iCprime,iSprime,kCprime,kSprime,x,jC,jS,y)] += weights[wnum] * ( (prop_re[prop_index(0,t,iC,iS,iCprime,iSprime,y,x)] * prop_re[prop_index(2,t,kC,kS,kCprime,kSprime,y,x)] - prop_im[prop_index(0,t,iC,iS,iCprime,iSprime,y,x)] * prop_im[prop_index(2,t,kC,kS,kCprime,kSprime,y,x)]) - (prop_re[prop_index(0,t,iC,iS,kCprime,kSprime,y,x)] * prop_re[prop_index(2,t,kC,kS,iCprime,iSprime,y,x)] - prop_im[prop_index(0,t,iC,iS,kCprime,kSprime,y,x)] * prop_im[prop_index(2,t,kC,kS,iCprime,iSprime,y,x)]) );
                           Q02_im[Q_index(t,iCprime,iSprime,kCprime,kSprime,x,jC,jS,y)] += weights[wnum] * ( (prop_re[prop_index(0,t,iC,iS,iCprime,iSprime,y,x)] * prop_im[prop_index(2,t,kC,kS,kCprime,kSprime,y,x)] + prop_im[prop_index(0,t,iC,iS,iCprime,iSprime,y,x)] * prop_re[prop_index(2,t,kC,kS,kCprime,kSprime,y,x)]) - (prop_re[prop_index(0,t,iC,iS,kCprime,kSprime,y,x)] * prop_im[prop_index(2,t,kC,kS,iCprime,iSprime,y,x)] + prop_im[prop_index(0,t,iC,iS,kCprime,kSprime,y,x)] * prop_re[prop_index(2,t,kC,kS,iCprime,iSprime,y,x)]) );
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
                                       prop_prod_re = Q02_re[Q_index(t,iCprime,iSprime,kCprime,kSprime,x1,jC,jS,y)] * prop_re[prop_index(1,t,jC,jS,jCprime,jSprime,y,x2)] - Q02_im[Q_index(t,iCprime,iSprime,kCprime,kSprime,x1,jC,jS,y)] * prop_im[prop_index(1,t,jC,jS,jCprime,jSprime,y,x2)];
                                       prop_prod_im = Q02_re[Q_index(t,iCprime,iSprime,kCprime,kSprime,x1,jC,jS,y)] * prop_im[prop_index(1,t,jC,jS,jCprime,jSprime,y,x2)] + Q02_im[Q_index(t,iCprime,iSprime,kCprime,kSprime,x1,jC,jS,y)] * prop_re[prop_index(1,t,jC,jS,jCprime,jSprime,y,x2)];
                                       for (m=0; m<Nsrc; m++) {
                                          Bsingle_re[Bsingle_index(t,iCprime,iSprime,kCprime,kSprime,x1,jCprime,jSprime,x2,m)] += psi_re[src_psi_index(y,m)] * prop_prod_re - psi_im[src_psi_index(y,m)] * prop_prod_im;
                                          Bsingle_im[Bsingle_index(t,iCprime,iSprime,kCprime,kSprime,x1,jCprime,jSprime,x2,m)] += psi_re[src_psi_index(y,m)] * prop_prod_im + psi_im[src_psi_index(y,m)] * prop_prod_re;
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
    const double* psi_im) {
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
                                 Bdouble_re[Bdouble_index(t,iCprime,iSprime,kCprime,kSprime,x1,jCprime,jSprime,x2,n)] = 0.0;
                                 Bdouble_im[Bdouble_index(t,iCprime,iSprime,kCprime,kSprime,x1,jCprime,jSprime,x2,n)] = 0.0;
                              }
                           }
                        }
                     }
                     for (jC=0; jC<Nc; jC++) {
                        for (jS=0; jS<Ns; jS++) {
                           for (y=0; y<Vsrc; y++) {
                              Q12_re[Q_index(t,iCprime,iSprime,kCprime,kSprime,x1,jC,jS,y)] = 0.0;
                              Q12_im[Q_index(t,iCprime,iSprime,kCprime,kSprime,x1,jC,jS,y)] = 0.0;
                              Q01_re[Q_index(t,iCprime,iSprime,kCprime,kSprime,x1,jC,jS,y)] = 0.0;
                              Q01_im[Q_index(t,iCprime,iSprime,kCprime,kSprime,x1,jC,jS,y)] = 0.0;
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
      iC = color_weights[src_weight_index(wnum,0)];
      iS = spin_weights[src_weight_index(wnum,0)];
      jC = color_weights[src_weight_index(wnum,1)];
      jS = spin_weights[src_weight_index(wnum,1)];
      kC = color_weights[src_weight_index(wnum,2)];
      kS = spin_weights[src_weight_index(wnum,2)];
      for (t=0; t<Nt; t++) {
         for (jCprime=0; jCprime<Nc; jCprime++) {
            for (jSprime=0; jSprime<Ns; jSprime++) {
               for (kCprime=0; kCprime<Nc; kCprime++) {
                  for (kSprime=0; kSprime<Ns; kSprime++) {
                     for (y=0; y<Vsrc; y++) {
                        for (x=0; x<Vsnk; x++) {
                           Q12_re[Q_index(t,jCprime,jSprime,kCprime,kSprime,x,iC,iS,y)] += weights[wnum] * (prop_re[prop_index(1,t,jC,jS,jCprime,jSprime,y,x)] * prop_re[prop_index(2,t,kC,kS,kCprime,kSprime,y,x)] - prop_im[prop_index(1,t,jC,jS,jCprime,jSprime,y,x)] * prop_im[prop_index(2,t,kC,kS,kCprime,kSprime,y,x)]);
                           Q12_im[Q_index(t,jCprime,jSprime,kCprime,kSprime,x,iC,iS,y)] += weights[wnum] * (prop_re[prop_index(1,t,jC,jS,jCprime,jSprime,y,x)] * prop_im[prop_index(2,t,kC,kS,kCprime,kSprime,y,x)] + prop_im[prop_index(1,t,jC,jS,jCprime,jSprime,y,x)] * prop_re[prop_index(2,t,kC,kS,kCprime,kSprime,y,x)]);
                           Q01_re[Q_index(t,jCprime,jSprime,kCprime,kSprime,x,kC,kS,y)] += weights[wnum] * (prop_re[prop_index(0,t,iC,iS,kCprime,kSprime,y,x)] * prop_re[prop_index(1,t,jC,jS,jCprime,jSprime,y,x)] - prop_im[prop_index(0,t,iC,iS,kCprime,kSprime,y,x)] * prop_im[prop_index(1,t,jC,jS,jCprime,jSprime,y,x)]);
                           Q01_im[Q_index(t,jCprime,jSprime,kCprime,kSprime,x,kC,kS,y)] += weights[wnum] * (prop_re[prop_index(0,t,iC,iS,kCprime,kSprime,y,x)] * prop_im[prop_index(1,t,jC,jS,jCprime,jSprime,y,x)] + prop_im[prop_index(0,t,iC,iS,kCprime,kSprime,y,x)] * prop_re[prop_index(1,t,jC,jS,jCprime,jSprime,y,x)]);
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
                                       prop_prod_re = Q12_re[Q_index(t,jCprime,jSprime,kCprime,kSprime,x1,iC,iS,y)] * prop_re[prop_index(0,t,iC,iS,iCprime,iSprime,y,x2)] - Q12_im[Q_index(t,jCprime,jSprime,kCprime,kSprime,x1,iC,iS,y)] * prop_im[prop_index(0,t,iC,iS,iCprime,iSprime,y,x2)];
                                       prop_prod_im = Q12_re[Q_index(t,jCprime,jSprime,kCprime,kSprime,x1,iC,iS,y)] * prop_im[prop_index(0,t,iC,iS,iCprime,iSprime,y,x2)] + Q12_im[Q_index(t,jCprime,jSprime,kCprime,kSprime,x1,iC,iS,y)] * prop_re[prop_index(0,t,iC,iS,iCprime,iSprime,y,x2)];
                                       prop_prod_re -= Q01_re[Q_index(t,jCprime,jSprime,kCprime,kSprime,x1,iC,iS,y)] * prop_re[prop_index(2,t,iC,iS,iCprime,iSprime,y,x2)] - Q01_im[Q_index(t,jCprime,jSprime,kCprime,kSprime,x1,iC,iS,y)] * prop_im[prop_index(2,t,iC,iS,iCprime,iSprime,y,x2)];
                                       prop_prod_im -= Q01_re[Q_index(t,jCprime,jSprime,kCprime,kSprime,x1,iC,iS,y)] * prop_im[prop_index(2,t,iC,iS,iCprime,iSprime,y,x2)] + Q01_im[Q_index(t,jCprime,jSprime,kCprime,kSprime,x1,iC,iS,y)] * prop_re[prop_index(2,t,iC,iS,iCprime,iSprime,y,x2)];
                                       for (n=0; n<Nsrc; n++) {
                                          Bdouble_re[Bdouble_index(t,iCprime,iSprime,kCprime,kSprime,x1,jCprime,jSprime,x2,n)] += psi_re[src_psi_index(y,n)] * prop_prod_re - psi_im[src_psi_index(y,n)] * prop_prod_im;
                                          Bdouble_im[Bdouble_index(t,iCprime,iSprime,kCprime,kSprime,x1,jCprime,jSprime,x2,n)] += psi_re[src_psi_index(y,n)] * prop_prod_im + psi_im[src_psi_index(y,n)] * prop_prod_re;
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
    const double* snk_psi_im) {
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
         snk_1[b] = perms[perm_index(nperm,Nq*b+0)] - 1;
         snk_2[b] = perms[perm_index(nperm,Nq*b+1)] - 1;
         snk_3[b] = perms[perm_index(nperm,Nq*b+2)] - 1;
         snk_1_b[b] = (snk_1[b] - snk_1[b] % Nq) / Nq;
         snk_2_b[b] = (snk_2[b] - snk_2[b] % Nq) / Nq;
         snk_3_b[b] = (snk_3[b] - snk_3[b] % Nq) / Nq;
         snk_1_nq[b] = snk_1[b] % Nq;
         snk_2_nq[b] = snk_2[b] % Nq;
         snk_3_nq[b] = snk_3[b] % Nq;
      }
      for (wnum=0; wnum< Nw2; wnum++) {
         iC1 = snk_color_weights[snk_weight_index(snk_1_b[0],wnum,snk_1_nq[0])];
         iS1 = snk_spin_weights[snk_weight_index(snk_1_b[0],wnum,snk_1_nq[0])];
         jC1 = snk_color_weights[snk_weight_index(snk_2_b[0],wnum,snk_2_nq[0])];
         jS1 = snk_spin_weights[snk_weight_index(snk_2_b[0],wnum,snk_2_nq[0])];
         kC1 = snk_color_weights[snk_weight_index(snk_3_b[0],wnum,snk_3_nq[0])];
         kS1 = snk_spin_weights[snk_weight_index(snk_3_b[0],wnum,snk_3_nq[0])];
         iC2 = snk_color_weights[snk_weight_index(snk_1_b[1],wnum,snk_1_nq[1])];
         iS2 = snk_spin_weights[snk_weight_index(snk_1_b[1],wnum,snk_1_nq[1])];
         jC2 = snk_color_weights[snk_weight_index(snk_2_b[1],wnum,snk_2_nq[1])];
         jS2 = snk_spin_weights[snk_weight_index(snk_2_b[1],wnum,snk_2_nq[1])];
         kC2 = snk_color_weights[snk_weight_index(snk_3_b[1],wnum,snk_3_nq[1])];
         kS2 = snk_spin_weights[snk_weight_index(snk_3_b[1],wnum,snk_3_nq[1])]; 
         for (t=0; t<Nt; t++) {
            for (x1=0; x1<Vsnk; x1++) {
               for (x2=0; x2<Vsnk; x2++) {
                  for (m=0; m<Nsrc; m++) {
                     term_re = sigs[nperm] * overall_weight * snk_weights[wnum];
                     term_im = 0.0;
                     for (b=0; b<Nb; b++) {
                        if ((snk_1_b[b] == 0) && (snk_2_b[b] == 0) && (snk_3_b[b] == 0)) {
                           new_term_re = term_re * B1_Blocal_re[Blocal_index(t,iC1,iS1,kC1,kS1,x1,jC1,jS1,m)] - term_im * B1_Blocal_im[Blocal_index(t,iC1,iS1,kC1,kS1,x1,jC1,jS1,m)];
                           new_term_im = term_re * B1_Blocal_im[Blocal_index(t,iC1,iS1,kC1,kS1,x1,jC1,jS1,m)] + term_im * B1_Blocal_re[Blocal_index(t,iC1,iS1,kC1,kS1,x1,jC1,jS1,m)];
                        }
                        else if ((snk_1_b[b] == 1) && (snk_2_b[b] == 1) && (snk_3_b[b] == 1)) {
                           new_term_re = term_re * B2_Blocal_re[Blocal_index(t,iC2,iS2,kC2,kS2,x2,jC2,jS2,m)] - term_im * B2_Blocal_im[Blocal_index(t,iC2,iS2,kC2,kS2,x2,jC2,jS2,m)];
                           new_term_im = term_re * B2_Blocal_im[Blocal_index(t,iC2,iS2,kC2,kS2,x2,jC2,jS2,m)] + term_im * B2_Blocal_re[Blocal_index(t,iC2,iS2,kC2,kS2,x2,jC2,jS2,m)];
                        }
                        else if ((snk_1_b[b] == 0) && (snk_3_b[b] == 0)) {
                           new_term_re = term_re * B1_Bsingle_re[Bsingle_index(t,iC1,iS1,kC1,kS1,x1,jC1,jS1,x2,m)] - term_im * B1_Bsingle_im[Bsingle_index(t,iC1,iS1,kC1,kS1,x1,jC1,jS1,x2,m)];
                           new_term_im = term_re * B1_Bsingle_im[Bsingle_index(t,iC1,iS1,kC1,kS1,x1,jC1,jS1,x2,m)] + term_im * B1_Bsingle_re[Bsingle_index(t,iC1,iS1,kC1,kS1,x1,jC1,jS1,x2,m)];
                        }
                        else if ((snk_1_b[b] == 1) && (snk_3_b[b] == 1)) {
                           new_term_re = term_re * B2_Bsingle_re[Bsingle_index(t,iC2,iS2,kC2,kS2,x2,jC2,jS2,x1,m)] - term_im * B2_Bsingle_im[Bsingle_index(t,iC2,iS2,kC2,kS2,x2,jC2,jS2,x1,m)];
                           new_term_im = term_re * B2_Bsingle_im[Bsingle_index(t,iC2,iS2,kC2,kS2,x2,jC2,jS2,x1,m)] + term_im * B2_Bsingle_re[Bsingle_index(t,iC2,iS2,kC2,kS2,x2,jC2,jS2,x1,m)];
                        }
                        else if (((snk_1_b[b] == 0) && (snk_2_b[b] == 0)) || ((snk_2_b[b] == 0) && (snk_3_b[b] == 0))) {
                           new_term_re = term_re * B1_Bdouble_re[Bdouble_index(t,iC1,iS1,kC1,kS1,x1,jC1,jS1,x2,m)] - term_im * B1_Bdouble_im[Bdouble_index(t,iC1,iS1,kC1,kS1,x1,jC1,jS1,x2,m)];
                           new_term_im = term_re * B1_Bdouble_im[Bdouble_index(t,iC1,iS1,kC1,kS1,x1,jC1,jS1,x2,m)] + term_im * B1_Bdouble_re[Bdouble_index(t,iC1,iS1,kC1,kS1,x1,jC1,jS1,x2,m)];
                        }
                        else if (((snk_1_b[b] == 1) && (snk_2_b[b] == 1)) || ((snk_2_b[b] == 1) && (snk_3_b[b] == 1))) {
                           new_term_re = term_re * B2_Bdouble_re[Bdouble_index(t,iC2,iS2,kC2,kS2,x2,jC2,jS2,x1,m)] - term_im * B2_Bdouble_im[Bdouble_index(t,iC2,iS2,kC2,kS2,x2,jC2,jS2,x1,m)];
                           new_term_im = term_re * B2_Bdouble_im[Bdouble_index(t,iC2,iS2,kC2,kS2,x2,jC2,jS2,x1,m)] + term_im * B2_Bdouble_re[Bdouble_index(t,iC2,iS2,kC2,kS2,x2,jC2,jS2,x1,m)];
                        }
                        term_re = new_term_re;
                        term_im = new_term_im;
                     }
                     for (n=0; n<Nsnk; n++) {
                        C_re[one_correlator_index(m,n,t)] += snk_psi_re[snk_psi_index(x1,x2,n)] * term_re - snk_psi_im[snk_psi_index(x1,x2,n)] * term_im;
                        C_im[one_correlator_index(m,n,t)] += snk_psi_re[snk_psi_index(x1,x2,n)] * term_im + snk_psi_im[snk_psi_index(x1,x2,n)] * term_re;
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
    const double* hex_snk_psi_im) {
   /* indices */
   int iC1,iS1,jC1,jS1,kC1,kS1,iC2,iS2,jC2,jS2,kC2,kS2,x,t,wnum,nperm,b,n,m,y;
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
         snk_1[b] = perms[perm_index(nperm,Nq*b+0)] - 1;
         snk_2[b] = perms[perm_index(nperm,Nq*b+1)] - 1;
         snk_3[b] = perms[perm_index(nperm,Nq*b+2)] - 1;
         snk_1_b[b] = (snk_1[b] - snk_1[b] % Nq) / Nq;
         snk_2_b[b] = (snk_2[b] - snk_2[b] % Nq) / Nq;
         snk_3_b[b] = (snk_3[b] - snk_3[b] % Nq) / Nq;
         snk_1_nq[b] = snk_1[b] % Nq;
         snk_2_nq[b] = snk_2[b] % Nq;
         snk_3_nq[b] = snk_3[b] % Nq;
      }
      for (wnum=0; wnum< Nw2; wnum++) {
         iC1 = snk_color_weights[snk_weight_index(snk_1_b[0],wnum,snk_1_nq[0])];
         iS1 = snk_spin_weights[snk_weight_index(snk_1_b[0],wnum,snk_1_nq[0])];
         jC1 = snk_color_weights[snk_weight_index(snk_2_b[0],wnum,snk_2_nq[0])];
         jS1 = snk_spin_weights[snk_weight_index(snk_2_b[0],wnum,snk_2_nq[0])];
         kC1 = snk_color_weights[snk_weight_index(snk_3_b[0],wnum,snk_3_nq[0])];
         kS1 = snk_spin_weights[snk_weight_index(snk_3_b[0],wnum,snk_3_nq[0])];
         iC2 = snk_color_weights[snk_weight_index(snk_1_b[1],wnum,snk_1_nq[1])];
         iS2 = snk_spin_weights[snk_weight_index(snk_1_b[1],wnum,snk_1_nq[1])];
         jC2 = snk_color_weights[snk_weight_index(snk_2_b[1],wnum,snk_2_nq[1])];
         jS2 = snk_spin_weights[snk_weight_index(snk_2_b[1],wnum,snk_2_nq[1])];
         kC2 = snk_color_weights[snk_weight_index(snk_3_b[1],wnum,snk_3_nq[1])];
         kS2 = snk_spin_weights[snk_weight_index(snk_3_b[1],wnum,snk_3_nq[1])]; 
         for (t=0; t<Nt; t++) {
            for (y=0; y<Vsnk; y++) {
               for (n=0; n<Nsrc; n++) {
                  term_re = sigs[nperm] * overall_weight * snk_weights[wnum] * (B1_Blocal_re[snk_Blocal_index(t,iC1,iS1,kC1,kS1,y,jC1,jS1,n)] * B2_Blocal_re[snk_Blocal_index(t,iC2,iS2,kC2,kS2,y,jC2,jS2,n)] - B1_Blocal_im[snk_Blocal_index(t,iC1,iS1,kC1,kS1,y,jC1,jS1,n)] * B2_Blocal_im[snk_Blocal_index(t,iC2,iS2,kC2,kS2,y,jC2,jS2,n)]);
                  term_im = sigs[nperm] * overall_weight * snk_weights[wnum] * (B1_Blocal_re[snk_Blocal_index(t,iC1,iS1,kC1,kS1,y,jC1,jS1,n)] * B2_Blocal_im[snk_Blocal_index(t,iC2,iS2,kC2,kS2,y,jC2,jS2,n)] + B1_Blocal_im[snk_Blocal_index(t,iC1,iS1,kC1,kS1,y,jC1,jS1,n)] * B2_Blocal_re[snk_Blocal_index(t,iC2,iS2,kC2,kS2,y,jC2,jS2,n)]);
                  for (m=0; m<NsnkHex; m++) {
                     C_re[one_hex_correlator_index(m,n,t)] += hex_snk_psi_re[hex_snk_psi_index(y,m)] * term_re - hex_snk_psi_im[hex_snk_psi_index(y,m)] * term_im;
                     C_im[one_hex_correlator_index(m,n,t)] += hex_snk_psi_re[hex_snk_psi_index(y,m)] * term_im + hex_snk_psi_im[hex_snk_psi_index(y,m)] * term_re;
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
    const double* hex_src_psi_im) {
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
         src_1[b] = perms[perm_index(nperm,Nq*b+0)] - 1;
         src_2[b] = perms[perm_index(nperm,Nq*b+1)] - 1;
         src_3[b] = perms[perm_index(nperm,Nq*b+2)] - 1;
         src_1_b[b] = (src_1[b] - src_1[b] % Nq) / Nq;
         src_2_b[b] = (src_2[b] - src_2[b] % Nq) / Nq;
         src_3_b[b] = (src_3[b] - src_3[b] % Nq) / Nq;
         src_1_nq[b] = src_1[b] % Nq;
         src_2_nq[b] = src_2[b] % Nq;
         src_3_nq[b] = src_3[b] % Nq;
      }
      for (wnum=0; wnum< Nw2; wnum++) {
         iC1 = src_color_weights[snk_weight_index(src_1_b[0],wnum,src_1_nq[0])];
         iS1 = src_spin_weights[snk_weight_index(src_1_b[0],wnum,src_1_nq[0])];
         jC1 = src_color_weights[snk_weight_index(src_2_b[0],wnum,src_2_nq[0])];
         jS1 = src_spin_weights[snk_weight_index(src_2_b[0],wnum,src_2_nq[0])];
         kC1 = src_color_weights[snk_weight_index(src_3_b[0],wnum,src_3_nq[0])];
         kS1 = src_spin_weights[snk_weight_index(src_3_b[0],wnum,src_3_nq[0])];
         iC2 = src_color_weights[snk_weight_index(src_1_b[1],wnum,src_1_nq[1])];
         iS2 = src_spin_weights[snk_weight_index(src_1_b[1],wnum,src_1_nq[1])];
         jC2 = src_color_weights[snk_weight_index(src_2_b[1],wnum,src_2_nq[1])];
         jS2 = src_spin_weights[snk_weight_index(src_2_b[1],wnum,src_2_nq[1])];
         kC2 = src_color_weights[snk_weight_index(src_3_b[1],wnum,src_3_nq[1])];
         kS2 = src_spin_weights[snk_weight_index(src_3_b[1],wnum,src_3_nq[1])]; 
         for (t=0; t<Nt; t++) {
            for (y=0; y<Vsrc; y++) {
               for (n=0; n<Nsnk; n++) {
                  term_re = sigs[nperm] * overall_weight * src_weights[wnum] * (B1_Blocal_re[snk_Blocal_index(t,iC1,iS1,kC1,kS1,y,jC1,jS1,n)] * B2_Blocal_re[snk_Blocal_index(t,iC2,iS2,kC2,kS2,y,jC2,jS2,n)] - B1_Blocal_im[snk_Blocal_index(t,iC1,iS1,kC1,kS1,y,jC1,jS1,n)] * B2_Blocal_im[snk_Blocal_index(t,iC2,iS2,kC2,kS2,y,jC2,jS2,n)]);
                  term_im = sigs[nperm] * overall_weight * src_weights[wnum] * (B1_Blocal_re[snk_Blocal_index(t,iC1,iS1,kC1,kS1,y,jC1,jS1,n)] * B2_Blocal_im[snk_Blocal_index(t,iC2,iS2,kC2,kS2,y,jC2,jS2,n)] + B1_Blocal_im[snk_Blocal_index(t,iC1,iS1,kC1,kS1,y,jC1,jS1,n)] * B2_Blocal_re[snk_Blocal_index(t,iC2,iS2,kC2,kS2,y,jC2,jS2,n)]);
                  for (m=0; m<NsrcHex; m++) {
                     C_re[one_correlator_index(m,n,t)] += hex_src_psi_re[hex_src_psi_index(y,m)] * term_re - hex_src_psi_im[hex_src_psi_index(y,m)] * term_im;
                     C_im[one_correlator_index(m,n,t)] += hex_src_psi_re[hex_src_psi_index(y,m)] * term_im + hex_src_psi_im[hex_src_psi_index(y,m)] * term_re;
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
    const double* hex_snk_psi_im) {
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
         snk_1[b] = perms[perm_index(nperm,Nq*b+0)] - 1;
         snk_2[b] = perms[perm_index(nperm,Nq*b+1)] - 1;
         snk_3[b] = perms[perm_index(nperm,Nq*b+2)] - 1;
         snk_1_b[b] = (snk_1[b] - snk_1[b] % Nq) / Nq;
         snk_2_b[b] = (snk_2[b] - snk_2[b] % Nq) / Nq;
         snk_3_b[b] = (snk_3[b] - snk_3[b] % Nq) / Nq;
         snk_1_nq[b] = snk_1[b] % Nq;
         snk_2_nq[b] = snk_2[b] % Nq;
         snk_3_nq[b] = snk_3[b] % Nq;
      }
      for (wnumprime=0; wnumprime< Nw2; wnumprime++) {
         iC1prime = snk_color_weights[snk_weight_index(snk_1_b[0],wnumprime,snk_1_nq[0])];
         iS1prime = snk_spin_weights[snk_weight_index(snk_1_b[0],wnumprime,snk_1_nq[0])];
         jC1prime = snk_color_weights[snk_weight_index(snk_2_b[0],wnumprime,snk_2_nq[0])];
         jS1prime = snk_spin_weights[snk_weight_index(snk_2_b[0],wnumprime,snk_2_nq[0])];
         kC1prime = snk_color_weights[snk_weight_index(snk_3_b[0],wnumprime,snk_3_nq[0])];
         kS1prime = snk_spin_weights[snk_weight_index(snk_3_b[0],wnumprime,snk_3_nq[0])];
         iC2prime = snk_color_weights[snk_weight_index(snk_1_b[1],wnumprime,snk_1_nq[1])];
         iS2prime = snk_spin_weights[snk_weight_index(snk_1_b[1],wnumprime,snk_1_nq[1])];
         jC2prime = snk_color_weights[snk_weight_index(snk_2_b[1],wnumprime,snk_2_nq[1])];
         jS2prime = snk_spin_weights[snk_weight_index(snk_2_b[1],wnumprime,snk_2_nq[1])];
         kC2prime = snk_color_weights[snk_weight_index(snk_3_b[1],wnumprime,snk_3_nq[1])];
         kS2prime = snk_spin_weights[snk_weight_index(snk_3_b[1],wnumprime,snk_3_nq[1])]; 
         for (t=0; t<Nt; t++) {
            for (y=0; y<Vsrc; y++) {
               for (x=0; x<Vsnk; x++) {
                  B1_prop_prod_re = 0;
                  B1_prop_prod_im = 0;
                  for (wnum=0; wnum<Nw; wnum++) {
                     iC1 = B1_src_color_weights[src_weight_index(wnum,0)];
                     iS1 = B1_src_spin_weights[src_weight_index(wnum,0)];
                     jC1 = B1_src_color_weights[src_weight_index(wnum,1)];
                     jS1 = B1_src_spin_weights[src_weight_index(wnum,1)];
                     kC1 = B1_src_color_weights[src_weight_index(wnum,2)];
                     kS1 = B1_src_spin_weights[src_weight_index(wnum,2)];
                     B1_prop_prod_02_re = B1_src_weights[wnum] * ( (B1_prop_re[prop_index(0,t,iC1,iS1,iC1prime,iS1prime,y,x)] * B1_prop_re[prop_index(2,t,kC1,kS1,kC1prime,kS1prime,y,x)] - B1_prop_im[prop_index(0,t,iC1,iS1,iC1prime,iS1prime,y,x)] * B1_prop_im[prop_index(2,t,kC1,kS1,kC1prime,kS1prime,y,x)]) - (B1_prop_re[prop_index(0,t,iC1,iS1,kC1prime,kS1prime,y,x)] * B1_prop_re[prop_index(2,t,kC1,kS1,iC1prime,iS1prime,y,x)] - B1_prop_im[prop_index(0,t,iC1,iS1,kC1prime,kS1prime,y,x)] * B1_prop_im[prop_index(2,t,kC1,kS1,iC1prime,iS1prime,y,x)]) );
                     B1_prop_prod_02_im = B1_src_weights[wnum] * ( (B1_prop_re[prop_index(0,t,iC1,iS1,iC1prime,iS1prime,y,x)] * B1_prop_im[prop_index(2,t,kC1,kS1,kC1prime,kS1prime,y,x)] + B1_prop_im[prop_index(0,t,iC1,iS1,iC1prime,iS1prime,y,x)] * B1_prop_re[prop_index(2,t,kC1,kS1,kC1prime,kS1prime,y,x)]) - (B1_prop_re[prop_index(0,t,iC1,iS1,kC1prime,kS1prime,y,x)] * B1_prop_im[prop_index(2,t,kC1,kS1,iC1prime,iS1prime,y,x)] + B1_prop_im[prop_index(0,t,iC1,iS1,kC1prime,kS1prime,y,x)] * B1_prop_re[prop_index(2,t,kC1,kS1,iC1prime,iS1prime,y,x)]) );
                     B1_prop_prod_re += B1_prop_prod_02_re * B1_prop_re[prop_index(1,t,jC1,jS1,jC1prime,jS1prime,y,x)] - B1_prop_prod_02_im * B1_prop_im[prop_index(1,t,jC1,jS1,jC1prime,jS1prime,y,x)];
                     B1_prop_prod_im += B1_prop_prod_02_re * B1_prop_im[prop_index(1,t,jC1,jS1,jC1prime,jS1prime,y,x)] + B1_prop_prod_02_im * B1_prop_re[prop_index(1,t,jC1,jS1,jC1prime,jS1prime,y,x)];
                  }
                  B2_prop_prod_re = 0;
                  B2_prop_prod_im = 0;
                  for (wnum=0; wnum<Nw; wnum++) {
                     iC2 = B2_src_color_weights[src_weight_index(wnum,0)];
                     iS2 = B2_src_spin_weights[src_weight_index(wnum,0)];
                     jC2 = B2_src_color_weights[src_weight_index(wnum,1)];
                     jS2 = B2_src_spin_weights[src_weight_index(wnum,1)];
                     kC2 = B2_src_color_weights[src_weight_index(wnum,2)];
                     kS2 = B2_src_spin_weights[src_weight_index(wnum,2)];
                     B2_prop_prod_02_re = B2_src_weights[wnum] * ( (B2_prop_re[prop_index(0,t,iC2,iS2,iC2prime,iS2prime,y,x)] * B2_prop_re[prop_index(2,t,kC2,kS2,kC2prime,kS2prime,y,x)] - B2_prop_im[prop_index(0,t,iC2,iS2,iC2prime,iS2prime,y,x)] * B2_prop_im[prop_index(2,t,kC2,kS2,kC2prime,kS2prime,y,x)]) - (B2_prop_re[prop_index(0,t,iC2,iS2,kC2prime,kS2prime,y,x)] * B2_prop_re[prop_index(2,t,kC2,kS2,iC2prime,iS2prime,y,x)] - B2_prop_im[prop_index(0,t,iC2,iS2,kC2prime,kS2prime,y,x)] * B2_prop_im[prop_index(2,t,kC2,kS2,iC2prime,iS2prime,y,x)]) );
                     B2_prop_prod_02_im = B2_src_weights[wnum] * ( (B2_prop_re[prop_index(0,t,iC2,iS2,iC2prime,iS2prime,y,x)] * B2_prop_im[prop_index(2,t,kC2,kS2,kC2prime,kS2prime,y,x)] + B2_prop_im[prop_index(0,t,iC2,iS2,iC2prime,iS2prime,y,x)] * B2_prop_re[prop_index(2,t,kC2,kS2,kC2prime,kS2prime,y,x)]) - (B2_prop_re[prop_index(0,t,iC2,iS2,kC2prime,kS2prime,y,x)] * B2_prop_im[prop_index(2,t,kC2,kS2,iC2prime,iS2prime,y,x)] + B2_prop_im[prop_index(0,t,iC2,iS2,kC2prime,kS2prime,y,x)] * B2_prop_re[prop_index(2,t,kC2,kS2,iC2prime,iS2prime,y,x)]) );
                     B2_prop_prod_re += B2_prop_prod_02_re * B2_prop_re[prop_index(1,t,jC2,jS2,jC2prime,jS2prime,y,x)] - B2_prop_prod_02_im * B2_prop_im[prop_index(1,t,jC2,jS2,jC2prime,jS2prime,y,x)];
                     B2_prop_prod_im += B2_prop_prod_02_re * B2_prop_im[prop_index(1,t,jC2,jS2,jC2prime,jS2prime,y,x)] + B2_prop_prod_02_im * B2_prop_re[prop_index(1,t,jC2,jS2,jC2prime,jS2prime,y,x)];
                  }
                  prop_prod_re = overall_weight * sigs[nperm] * snk_weights[wnumprime] * ( B1_prop_prod_re * B2_prop_prod_re - B1_prop_prod_im * B2_prop_prod_im );
                  prop_prod_im = overall_weight * sigs[nperm] * snk_weights[wnumprime] * ( B1_prop_prod_re * B2_prop_prod_im + B1_prop_prod_im * B2_prop_prod_re );
                  for (m=0; m<NsrcHex; m++) {
                     new_prop_prod_re = hex_src_psi_re[hex_src_psi_index(y,m)] * prop_prod_re - hex_src_psi_im[hex_src_psi_index(y,m)] * prop_prod_im;
                     new_prop_prod_im = hex_src_psi_re[hex_src_psi_index(y,m)] * prop_prod_im + hex_src_psi_im[hex_src_psi_index(y,m)] * prop_prod_re;
                     for (n=0; n<NsnkHex; n++) {
                        C_re[one_hex_correlator_index(m,n,t)] += hex_snk_psi_re[hex_snk_psi_index(x,n)] * new_prop_prod_re - hex_snk_psi_im[hex_snk_psi_index(x,n)] * new_prop_prod_im;
                        C_im[one_hex_correlator_index(m,n,t)] += hex_snk_psi_re[hex_snk_psi_index(x,n)] * new_prop_prod_im + hex_snk_psi_im[hex_snk_psi_index(x,n)] * new_prop_prod_re;
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
    const int snk_entangled) {
   /* indices */
   int nB1, nB2, nq, n, m, t;
   double overall_weight;
   double* hex_full_src_psi_re = malloc(Vsrc * Nsrc * sizeof (double));
   double* hex_full_src_psi_im = malloc(Vsrc * Nsrc * sizeof (double));
   double* hex_full_snk_psi_re = malloc(Vsrc * Nsrc * sizeof (double));
   double* hex_full_snk_psi_im = malloc(Vsrc * Nsrc * sizeof (double));
   /* create blocks */
   double* B1_Blocal_r1_re = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Nsrc * sizeof (double));
   double* B1_Blocal_r1_im = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Nsrc * sizeof (double));
   double* B1_Blocal_r2_re = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Nsrc * sizeof (double));
   double* B1_Blocal_r2_im = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Nsrc * sizeof (double));
   double* B1_Bsingle_r1_re = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Vsnk * Nsrc * sizeof (double));
   double* B1_Bsingle_r1_im = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Vsnk * Nsrc * sizeof (double));
   double* B1_Bsingle_r2_re = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Vsnk * Nsrc * sizeof (double));
   double* B1_Bsingle_r2_im = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Vsnk * Nsrc * sizeof (double));
   double* B1_Bdouble_r1_re = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Vsnk * Nsrc * sizeof (double));
   double* B1_Bdouble_r1_im = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Vsnk * Nsrc * sizeof (double));
   double* B1_Bdouble_r2_re = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Vsnk * Nsrc * sizeof (double));
   double* B1_Bdouble_r2_im = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Vsnk * Nsrc * sizeof (double));
   make_local_block(B1_Blocal_r1_re, B1_Blocal_r1_im, B1_prop_re, B1_prop_im, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, src_psi_B1_re, src_psi_B1_im);
   make_local_block(B1_Blocal_r2_re, B1_Blocal_r2_im, B1_prop_re, B1_prop_im, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, src_psi_B1_re, src_psi_B1_im);
   make_single_block(B1_Bsingle_r1_re, B1_Bsingle_r1_im, B1_prop_re, B1_prop_im, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, src_psi_B1_re, src_psi_B1_im);
   make_single_block(B1_Bsingle_r2_re, B1_Bsingle_r2_im, B1_prop_re, B1_prop_im, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, src_psi_B1_re, src_psi_B1_im);
   make_double_block(B1_Bdouble_r1_re, B1_Bdouble_r1_im, B1_prop_re, B1_prop_im, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, src_psi_B1_re, src_psi_B1_im);
   make_double_block(B1_Bdouble_r2_re, B1_Bdouble_r2_im, B1_prop_re, B1_prop_im, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, src_psi_B1_re, src_psi_B1_im);
   double* B2_Blocal_r1_re = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Nsrc * sizeof (double));
   double* B2_Blocal_r1_im = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Nsrc * sizeof (double));
   double* B2_Blocal_r2_re = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Nsrc * sizeof (double));
   double* B2_Blocal_r2_im = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Nsrc * sizeof (double));
   double* B2_Bsingle_r1_re = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Vsnk * Nsrc * sizeof (double));
   double* B2_Bsingle_r1_im = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Vsnk * Nsrc * sizeof (double));
   double* B2_Bsingle_r2_re = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Vsnk * Nsrc * sizeof (double));
   double* B2_Bsingle_r2_im = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Vsnk * Nsrc * sizeof (double));
   double* B2_Bdouble_r1_re = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Vsnk * Nsrc * sizeof (double));
   double* B2_Bdouble_r1_im = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Vsnk * Nsrc * sizeof (double));
   double* B2_Bdouble_r2_re = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Vsnk * Nsrc * sizeof (double));
   double* B2_Bdouble_r2_im = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Vsnk * Nsrc * sizeof (double));
   make_local_block(B2_Blocal_r1_re, B2_Blocal_r1_im, B2_prop_re, B2_prop_im, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, src_psi_B2_re, src_psi_B2_im);
   make_local_block(B2_Blocal_r2_re, B2_Blocal_r2_im, B2_prop_re, B2_prop_im, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, src_psi_B2_re, src_psi_B2_im);
   make_single_block(B2_Bsingle_r1_re, B2_Bsingle_r1_im, B2_prop_re, B2_prop_im, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, src_psi_B2_re, src_psi_B2_im);
   make_single_block(B2_Bsingle_r2_re, B2_Bsingle_r2_im, B2_prop_re, B2_prop_im, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, src_psi_B2_re, src_psi_B2_im);
   make_double_block(B2_Bdouble_r1_re, B2_Bdouble_r1_im, B2_prop_re, B2_prop_im, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, src_psi_B2_re, src_psi_B2_im);
   make_double_block(B2_Bdouble_r2_re, B2_Bdouble_r2_im, B2_prop_re, B2_prop_im, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, src_psi_B2_re, src_psi_B2_im);
   /* hold results for two nucleon correlators */
   double* BB_0_re = malloc(Nsrc * Nsnk * Nt * sizeof (double));
   double* BB_0_im = malloc(Nsrc * Nsnk * Nt * sizeof (double));
   double* BB_r1_re = malloc(Nsrc * Nsnk * Nt * sizeof (double));
   double* BB_r1_im = malloc(Nsrc * Nsnk * Nt * sizeof (double));
   double* BB_r2_re = malloc(Nsrc * Nsnk * Nt * sizeof (double));
   double* BB_r2_im = malloc(Nsrc * Nsnk * Nt * sizeof (double));
   double* BB_r3_re = malloc(Nsrc * Nsnk * Nt * sizeof (double));
   double* BB_r3_im = malloc(Nsrc * Nsnk * Nt * sizeof (double));
   double* H_0_re = malloc(NsrcHex * NsnkHex * Nt * sizeof (double));
   double* H_0_im = malloc(NsrcHex * NsnkHex * Nt * sizeof (double));
   double* H_r1_re = malloc(NsrcHex * NsnkHex * Nt * sizeof (double));
   double* H_r1_im = malloc(NsrcHex * NsnkHex * Nt * sizeof (double));
   double* H_r2_re = malloc(NsrcHex * NsnkHex * Nt * sizeof (double));
   double* H_r2_im = malloc(NsrcHex * NsnkHex * Nt * sizeof (double));
   double* H_r3_re = malloc(NsrcHex * NsnkHex * Nt * sizeof (double));
   double* H_r3_im = malloc(NsrcHex * NsnkHex * Nt * sizeof (double));
   double* BB_H_0_re = malloc(Nsrc * NsnkHex * Nt * sizeof (double));
   double* BB_H_0_im = malloc(Nsrc * NsnkHex * Nt * sizeof (double));
   double* BB_H_r1_re = malloc(Nsrc * NsnkHex * Nt * sizeof (double));
   double* BB_H_r1_im = malloc(Nsrc * NsnkHex * Nt * sizeof (double));
   double* BB_H_r2_re = malloc(Nsrc * NsnkHex * Nt * sizeof (double));
   double* BB_H_r2_im = malloc(Nsrc * NsnkHex * Nt * sizeof (double));
   double* BB_H_r3_re = malloc(Nsrc * NsnkHex * Nt * sizeof (double));
   double* BB_H_r3_im = malloc(Nsrc * NsnkHex * Nt * sizeof (double));
   double* H_BB_0_re = malloc(NsrcHex * Nsnk * Nt * sizeof (double));
   double* H_BB_0_im = malloc(NsrcHex * Nsnk * Nt * sizeof (double));
   double* H_BB_r1_re = malloc(NsrcHex * Nsnk * Nt * sizeof (double));
   double* H_BB_r1_im = malloc(NsrcHex * Nsnk * Nt * sizeof (double));
   double* H_BB_r2_re = malloc(NsrcHex * Nsnk * Nt * sizeof (double));
   double* H_BB_r2_im = malloc(NsrcHex * Nsnk * Nt * sizeof (double));
   double* H_BB_r3_re = malloc(NsrcHex * Nsnk * Nt * sizeof (double));
   double* H_BB_r3_im = malloc(NsrcHex * Nsnk * Nt * sizeof (double));
   for (m=0; m<Nsrc; m++) {
      for (n=0; n<Nsnk; n++) {
         for (t=0; t<Nt; t++) {
            BB_0_re[one_correlator_index(m,n,t)] = 0.0;
            BB_0_im[one_correlator_index(m,n,t)] = 0.0;
            BB_r1_re[one_correlator_index(m,n,t)] = 0.0;
            BB_r1_im[one_correlator_index(m,n,t)] = 0.0;
            BB_r2_re[one_correlator_index(m,n,t)] = 0.0;
            BB_r2_im[one_correlator_index(m,n,t)] = 0.0;
            BB_r3_re[one_correlator_index(m,n,t)] = 0.0;
            BB_r3_im[one_correlator_index(m,n,t)] = 0.0;
            H_0_re[one_hex_correlator_index(m,n,t)] = 0.0;
            H_0_im[one_hex_correlator_index(m,n,t)] = 0.0;
            H_r1_re[one_hex_correlator_index(m,n,t)] = 0.0;
            H_r1_im[one_hex_correlator_index(m,n,t)] = 0.0;
            H_r2_re[one_hex_correlator_index(m,n,t)] = 0.0;
            H_r2_im[one_hex_correlator_index(m,n,t)] = 0.0;
            H_r3_re[one_hex_correlator_index(m,n,t)] = 0.0;
            H_r3_im[one_hex_correlator_index(m,n,t)] = 0.0;
            BB_H_0_re[one_hex_correlator_index(m,n,t)] = 0.0;
            BB_H_0_im[one_hex_correlator_index(m,n,t)] = 0.0;
            BB_H_r1_re[one_hex_correlator_index(m,n,t)] = 0.0;
            BB_H_r1_im[one_hex_correlator_index(m,n,t)] = 0.0;
            BB_H_r2_re[one_hex_correlator_index(m,n,t)] = 0.0;
            BB_H_r2_im[one_hex_correlator_index(m,n,t)] = 0.0;
            BB_H_r3_re[one_hex_correlator_index(m,n,t)] = 0.0;
            BB_H_r3_im[one_hex_correlator_index(m,n,t)] = 0.0;
            H_BB_0_re[one_correlator_index(m,n,t)] = 0.0;
            H_BB_0_im[one_correlator_index(m,n,t)] = 0.0;
            H_BB_r1_re[one_correlator_index(m,n,t)] = 0.0;
            H_BB_r1_im[one_correlator_index(m,n,t)] = 0.0;
            H_BB_r2_re[one_correlator_index(m,n,t)] = 0.0;
            H_BB_r2_im[one_correlator_index(m,n,t)] = 0.0;
            H_BB_r3_re[one_correlator_index(m,n,t)] = 0.0;
            H_BB_r3_im[one_correlator_index(m,n,t)] = 0.0;
         }
      }
   }
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
            /* A1g */
            snk_color_weights_1[snk_weight_index(0,nB1+Nw*nB2,nq)] = src_color_weights_r1[src_weight_index(nB1,nq)];
            snk_spin_weights_1[snk_weight_index(0,nB1+Nw*nB2,nq)] = src_spin_weights_r1[src_weight_index(nB1,nq)];
            snk_color_weights_1[snk_weight_index(1,nB1+Nw*nB2,nq)] = src_color_weights_r2[src_weight_index(nB2,nq)];
            snk_spin_weights_1[snk_weight_index(1,nB1+Nw*nB2,nq)] = src_spin_weights_r2[src_weight_index(nB2,nq)];
            snk_color_weights_2[snk_weight_index(0,nB1+Nw*nB2,nq)] = src_color_weights_r2[src_weight_index(nB1,nq)];
            snk_spin_weights_2[snk_weight_index(0,nB1+Nw*nB2,nq)] = src_spin_weights_r2[src_weight_index(nB1,nq)];
            snk_color_weights_2[snk_weight_index(1,nB1+Nw*nB2,nq)] = src_color_weights_r1[src_weight_index(nB2,nq)];
            snk_spin_weights_2[snk_weight_index(1,nB1+Nw*nB2,nq)] = src_spin_weights_r1[src_weight_index(nB2,nq)];
            /* T1g_r1 */
            snk_color_weights_r1[snk_weight_index(0,nB1+Nw*nB2,nq)] = src_color_weights_r1[src_weight_index(nB1,nq)];
            snk_spin_weights_r1[snk_weight_index(0,nB1+Nw*nB2,nq)] = src_spin_weights_r1[src_weight_index(nB1,nq)];
            snk_color_weights_r1[snk_weight_index(1,nB1+Nw*nB2,nq)] = src_color_weights_r1[src_weight_index(nB2,nq)];
            snk_spin_weights_r1[snk_weight_index(1,nB1+Nw*nB2,nq)] = src_spin_weights_r1[src_weight_index(nB2,nq)];
            /* T1g_r2 */
            snk_color_weights_r2_1[snk_weight_index(0,nB1+Nw*nB2,nq)] = src_color_weights_r1[src_weight_index(nB1,nq)];
            snk_spin_weights_r2_1[snk_weight_index(0,nB1+Nw*nB2,nq)] = src_spin_weights_r1[src_weight_index(nB1,nq)];
            snk_color_weights_r2_1[snk_weight_index(1,nB1+Nw*nB2,nq)] = src_color_weights_r2[src_weight_index(nB2,nq)];
            snk_spin_weights_r2_1[snk_weight_index(1,nB1+Nw*nB2,nq)] = src_spin_weights_r2[src_weight_index(nB2,nq)];
            snk_color_weights_r2_2[snk_weight_index(0,nB1+Nw*nB2,nq)] = src_color_weights_r2[src_weight_index(nB1,nq)];
            snk_spin_weights_r2_2[snk_weight_index(0,nB1+Nw*nB2,nq)] = src_spin_weights_r2[src_weight_index(nB1,nq)];
            snk_color_weights_r2_2[snk_weight_index(1,nB1+Nw*nB2,nq)] = src_color_weights_r1[src_weight_index(nB2,nq)];
            snk_spin_weights_r2_2[snk_weight_index(1,nB1+Nw*nB2,nq)] = src_spin_weights_r1[src_weight_index(nB2,nq)];
            /* T1g_r3 */
            snk_color_weights_r3[snk_weight_index(0,nB1+Nw*nB2,nq)] = src_color_weights_r2[src_weight_index(nB1,nq)];
            snk_spin_weights_r3[snk_weight_index(0,nB1+Nw*nB2,nq)] = src_spin_weights_r2[src_weight_index(nB1,nq)];
            snk_color_weights_r3[snk_weight_index(1,nB1+Nw*nB2,nq)] = src_color_weights_r2[src_weight_index(nB2,nq)];
            snk_spin_weights_r3[snk_weight_index(1,nB1+Nw*nB2,nq)] = src_spin_weights_r2[src_weight_index(nB2,nq)];
         }
      }
   }
   if (space_symmetric == 0) {
      overall_weight = 1.0;
   }
   else {
      overall_weight = 1.0/2.0;
   } 
   /* compute two nucleon correlators from blocks */
   /* BB_BB */
   make_dibaryon_correlator(BB_0_re, BB_0_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B1_Bsingle_r1_re, B1_Bsingle_r1_im, B1_Bdouble_r1_re, B1_Bdouble_r1_im, B2_Blocal_r2_re, B2_Blocal_r2_im, B2_Bsingle_r2_re, B2_Bsingle_r2_im, B2_Bdouble_r2_re, B2_Bdouble_r2_im, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_1, snk_spin_weights_1, snk_weights_1, snk_psi_re, snk_psi_im);
   make_dibaryon_correlator(BB_0_re, BB_0_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B1_Bsingle_r1_re, B1_Bsingle_r1_im, B1_Bdouble_r1_re, B1_Bdouble_r1_im, B2_Blocal_r2_re, B2_Blocal_r2_im, B2_Bsingle_r2_re, B2_Bsingle_r2_im, B2_Bdouble_r2_re, B2_Bdouble_r2_im, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_2, snk_spin_weights_2, snk_weights_2, snk_psi_re, snk_psi_im);
   make_dibaryon_correlator(BB_0_re, BB_0_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B1_Bsingle_r2_re, B1_Bsingle_r2_im, B1_Bdouble_r2_re, B1_Bdouble_r2_im, B2_Blocal_r1_re, B2_Blocal_r1_im, B2_Bsingle_r1_re, B2_Bsingle_r1_im, B2_Bdouble_r1_re, B2_Bdouble_r1_im, perms, sigs, -1.0*overall_weight/sqrt(2.0), snk_color_weights_1, snk_spin_weights_1, snk_weights_1, snk_psi_re, snk_psi_im);
   make_dibaryon_correlator(BB_0_re, BB_0_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B1_Bsingle_r2_re, B1_Bsingle_r2_im, B1_Bdouble_r2_re, B1_Bdouble_r2_im, B2_Blocal_r1_re, B2_Blocal_r1_im, B2_Bsingle_r1_re, B2_Bsingle_r1_im, B2_Bdouble_r1_re, B2_Bdouble_r1_im, perms, sigs, -1.0*overall_weight/sqrt(2.0), snk_color_weights_2, snk_spin_weights_2, snk_weights_2, snk_psi_re, snk_psi_im);
   make_dibaryon_correlator(BB_r1_re, BB_r1_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B1_Bsingle_r1_re, B1_Bsingle_r1_im, B1_Bdouble_r1_re, B1_Bdouble_r1_im, B2_Blocal_r1_re, B2_Blocal_r1_im, B2_Bsingle_r1_re, B2_Bsingle_r1_im, B2_Bdouble_r1_re, B2_Bdouble_r1_im, perms, sigs, overall_weight, snk_color_weights_r1, snk_spin_weights_r1, snk_weights_r1, snk_psi_re, snk_psi_im);
   make_dibaryon_correlator(BB_r2_re, BB_r2_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B1_Bsingle_r1_re, B1_Bsingle_r1_im, B1_Bdouble_r1_re, B1_Bdouble_r1_im, B2_Blocal_r2_re, B2_Blocal_r2_im, B2_Bsingle_r2_re, B2_Bsingle_r2_im, B2_Bdouble_r2_re, B2_Bdouble_r2_im, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_r2_1, snk_spin_weights_r2_1, snk_weights_r2_1, snk_psi_re, snk_psi_im);
   make_dibaryon_correlator(BB_r2_re, BB_r2_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B1_Bsingle_r1_re, B1_Bsingle_r1_im, B1_Bdouble_r1_re, B1_Bdouble_r1_im, B2_Blocal_r2_re, B2_Blocal_r2_im, B2_Bsingle_r2_re, B2_Bsingle_r2_im, B2_Bdouble_r2_re, B2_Bdouble_r2_im, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_r2_2, snk_spin_weights_r2_2, snk_weights_r2_2, snk_psi_re, snk_psi_im);
   make_dibaryon_correlator(BB_r2_re, BB_r2_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B1_Bsingle_r2_re, B1_Bsingle_r2_im, B1_Bdouble_r2_re, B1_Bdouble_r2_im, B2_Blocal_r1_re, B2_Blocal_r1_im, B2_Bsingle_r1_re, B2_Bsingle_r1_im, B2_Bdouble_r1_re, B2_Bdouble_r1_im, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_r2_1, snk_spin_weights_r2_1, snk_weights_r2_1, snk_psi_re, snk_psi_im);
   make_dibaryon_correlator(BB_r2_re, BB_r2_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B1_Bsingle_r2_re, B1_Bsingle_r2_im, B1_Bdouble_r2_re, B1_Bdouble_r2_im, B2_Blocal_r1_re, B2_Blocal_r1_im, B2_Bsingle_r1_re, B2_Bsingle_r1_im, B2_Bdouble_r1_re, B2_Bdouble_r1_im, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_r2_2, snk_spin_weights_r2_2, snk_weights_r2_2, snk_psi_re, snk_psi_im);
   make_dibaryon_correlator(BB_r3_re, BB_r3_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B1_Bsingle_r2_re, B1_Bsingle_r2_im, B1_Bdouble_r2_re, B1_Bdouble_r2_im, B2_Blocal_r2_re, B2_Blocal_r2_im, B2_Bsingle_r2_re, B2_Bsingle_r2_im, B2_Bdouble_r2_re, B2_Bdouble_r2_im, perms, sigs, overall_weight, snk_color_weights_r3, snk_spin_weights_r3, snk_weights_r3, snk_psi_re, snk_psi_im);
   if (space_symmetric != 0) {
      make_dibaryon_correlator(BB_0_re, BB_0_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B1_Bsingle_r1_re, B1_Bsingle_r1_im, B1_Bdouble_r1_re, B1_Bdouble_r1_im, B2_Blocal_r2_re, B2_Blocal_r2_im, B2_Bsingle_r2_re, B2_Bsingle_r2_im, B2_Bdouble_r2_re, B2_Bdouble_r2_im, perms, sigs, space_symmetric*overall_weight/sqrt(2.0), snk_color_weights_1, snk_spin_weights_1, snk_weights_1, snk_psi_re, snk_psi_im);
      make_dibaryon_correlator(BB_0_re, BB_0_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B1_Bsingle_r1_re, B1_Bsingle_r1_im, B1_Bdouble_r1_re, B1_Bdouble_r1_im, B2_Blocal_r2_re, B2_Blocal_r2_im, B2_Bsingle_r2_re, B2_Bsingle_r2_im, B2_Bdouble_r2_re, B2_Bdouble_r2_im, perms, sigs, space_symmetric*overall_weight/sqrt(2.0), snk_color_weights_2, snk_spin_weights_2, snk_weights_2, snk_psi_re, snk_psi_im);
      make_dibaryon_correlator(BB_0_re, BB_0_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B1_Bsingle_r2_re, B1_Bsingle_r2_im, B1_Bdouble_r2_re, B1_Bdouble_r2_im, B2_Blocal_r1_re, B2_Blocal_r1_im, B2_Bsingle_r1_re, B2_Bsingle_r1_im, B2_Bdouble_r1_re, B2_Bdouble_r1_im, perms, sigs, -1.0*space_symmetric*overall_weight/sqrt(2.0), snk_color_weights_1, snk_spin_weights_1, snk_weights_1, snk_psi_re, snk_psi_im);
      make_dibaryon_correlator(BB_0_re, BB_0_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B1_Bsingle_r2_re, B1_Bsingle_r2_im, B1_Bdouble_r2_re, B1_Bdouble_r2_im, B2_Blocal_r1_re, B2_Blocal_r1_im, B2_Bsingle_r1_re, B2_Bsingle_r1_im, B2_Bdouble_r1_re, B2_Bdouble_r1_im, perms, sigs, -1.0*space_symmetric*overall_weight/sqrt(2.0), snk_color_weights_2, snk_spin_weights_2, snk_weights_2, snk_psi_re, snk_psi_im);
      make_dibaryon_correlator(BB_r1_re, BB_r1_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B1_Bsingle_r1_re, B1_Bsingle_r1_im, B1_Bdouble_r1_re, B1_Bdouble_r1_im, B2_Blocal_r1_re, B2_Blocal_r1_im, B2_Bsingle_r1_re, B2_Bsingle_r1_im, B2_Bdouble_r1_re, B2_Bdouble_r1_im, perms, sigs, space_symmetric*overall_weight, snk_color_weights_r1, snk_spin_weights_r1, snk_weights_r1, snk_psi_re, snk_psi_im);
      make_dibaryon_correlator(BB_r2_re, BB_r2_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B1_Bsingle_r1_re, B1_Bsingle_r1_im, B1_Bdouble_r1_re, B1_Bdouble_r1_im, B2_Blocal_r2_re, B2_Blocal_r2_im, B2_Bsingle_r2_re, B2_Bsingle_r2_im, B2_Bdouble_r2_re, B2_Bdouble_r2_im, perms, sigs, space_symmetric*overall_weight/sqrt(2.0), snk_color_weights_r2_1, snk_spin_weights_r2_1, snk_weights_r2_1, snk_psi_re, snk_psi_im);
      make_dibaryon_correlator(BB_r2_re, BB_r2_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B1_Bsingle_r1_re, B1_Bsingle_r1_im, B1_Bdouble_r1_re, B1_Bdouble_r1_im, B2_Blocal_r2_re, B2_Blocal_r2_im, B2_Bsingle_r2_re, B2_Bsingle_r2_im, B2_Bdouble_r2_re, B2_Bdouble_r2_im, perms, sigs, space_symmetric*overall_weight/sqrt(2.0), snk_color_weights_r2_2, snk_spin_weights_r2_2, snk_weights_r2_2, snk_psi_re, snk_psi_im);
      make_dibaryon_correlator(BB_r2_re, BB_r2_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B1_Bsingle_r2_re, B1_Bsingle_r2_im, B1_Bdouble_r2_re, B1_Bdouble_r2_im, B2_Blocal_r1_re, B2_Blocal_r1_im, B2_Bsingle_r1_re, B2_Bsingle_r1_im, B2_Bdouble_r1_re, B2_Bdouble_r1_im, perms, sigs, space_symmetric*overall_weight/sqrt(2.0), snk_color_weights_r2_1, snk_spin_weights_r2_1, snk_weights_r2_1, snk_psi_re, snk_psi_im);
      make_dibaryon_correlator(BB_r2_re, BB_r2_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B1_Bsingle_r2_re, B1_Bsingle_r2_im, B1_Bdouble_r2_re, B1_Bdouble_r2_im, B2_Blocal_r1_re, B2_Blocal_r1_im, B2_Bsingle_r1_re, B2_Bsingle_r1_im, B2_Bdouble_r1_re, B2_Bdouble_r1_im, perms, sigs, space_symmetric*overall_weight/sqrt(2.0), snk_color_weights_r2_2, snk_spin_weights_r2_2, snk_weights_r2_2, snk_psi_re, snk_psi_im);
      make_dibaryon_correlator(BB_r3_re, BB_r3_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B1_Bsingle_r2_re, B1_Bsingle_r2_im, B1_Bdouble_r2_re, B1_Bdouble_r2_im, B2_Blocal_r2_re, B2_Blocal_r2_im, B2_Bsingle_r2_re, B2_Bsingle_r2_im, B2_Bdouble_r2_re, B2_Bdouble_r2_im, perms, sigs, space_symmetric*overall_weight, snk_color_weights_r3, snk_spin_weights_r3, snk_weights_r3, snk_psi_re, snk_psi_im);
   }
   for (m=0; m<Nsrc; m++) {
      for (n=0; n<Nsnk; n++) {
         for (t=0; t<Nt; t++) {
            C_re[correlator_index(0,m,n,t)] = BB_0_re[one_correlator_index(m,n,t)];
            C_im[correlator_index(0,m,n,t)] = BB_0_im[one_correlator_index(m,n,t)];
            C_re[correlator_index(1,m,n,t)] = BB_r1_re[one_correlator_index(m,n,t)];
            C_im[correlator_index(1,m,n,t)] = BB_r1_im[one_correlator_index(m,n,t)];
            C_re[correlator_index(2,m,n,t)] = BB_r2_re[one_correlator_index(m,n,t)];
            C_im[correlator_index(2,m,n,t)] = BB_r2_im[one_correlator_index(m,n,t)];
            C_re[correlator_index(3,m,n,t)] = BB_r3_re[one_correlator_index(m,n,t)];
            C_im[correlator_index(3,m,n,t)] = BB_r3_im[one_correlator_index(m,n,t)];
         }
      }
   }
   /* H_H */
   make_hex_correlator(H_0_re, H_0_im, B1_prop_re, B1_prop_im, B2_prop_re, B2_prop_im, perms, sigs, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, overall_weight/sqrt(2.0), snk_color_weights_1, snk_spin_weights_1, snk_weights_1, hex_src_psi_re, hex_src_psi_im, hex_snk_psi_re, hex_snk_psi_im);
   make_hex_correlator(H_0_re, H_0_im, B1_prop_re, B1_prop_im, B2_prop_re, B2_prop_im, perms, sigs, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, -1.0*overall_weight/sqrt(2.0), snk_color_weights_1, snk_spin_weights_1, snk_weights_1, hex_src_psi_re, hex_src_psi_im, hex_snk_psi_re, hex_snk_psi_im);
   make_hex_correlator(H_0_re, H_0_im, B1_prop_re, B1_prop_im, B2_prop_re, B2_prop_im, perms, sigs, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, overall_weight/sqrt(2.0), snk_color_weights_2, snk_spin_weights_2, snk_weights_2, hex_src_psi_re, hex_src_psi_im, hex_snk_psi_re, hex_snk_psi_im);
   make_hex_correlator(H_0_re, H_0_im, B1_prop_re, B1_prop_im, B2_prop_re, B2_prop_im, perms, sigs, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, -1.0*overall_weight/sqrt(2.0), snk_color_weights_2, snk_spin_weights_2, snk_weights_2, hex_src_psi_re, hex_src_psi_im, hex_snk_psi_re, hex_snk_psi_im);
   make_hex_correlator(H_r1_re, H_r1_im, B1_prop_re, B1_prop_im, B2_prop_re, B2_prop_im, perms, sigs, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, overall_weight, snk_color_weights_r1, snk_spin_weights_r1, snk_weights_r1, hex_src_psi_re, hex_src_psi_im, hex_snk_psi_re, hex_snk_psi_im);
   make_hex_correlator(H_r2_re, H_r2_im, B1_prop_re, B1_prop_im, B2_prop_re, B2_prop_im, perms, sigs, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, overall_weight/sqrt(2.0), snk_color_weights_r2_1, snk_spin_weights_r2_1, snk_weights_r2_1, hex_src_psi_re, hex_src_psi_im, hex_snk_psi_re, hex_snk_psi_im);
   make_hex_correlator(H_r2_re, H_r2_im, B1_prop_re, B1_prop_im, B2_prop_re, B2_prop_im, perms, sigs, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, overall_weight/sqrt(2.0), snk_color_weights_r2_1, snk_spin_weights_r2_1, snk_weights_r2_1, hex_src_psi_re, hex_src_psi_im, hex_snk_psi_re, hex_snk_psi_im);
   make_hex_correlator(H_r2_re, H_r2_im, B1_prop_re, B1_prop_im, B2_prop_re, B2_prop_im, perms, sigs, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, overall_weight/sqrt(2.0), snk_color_weights_r2_2, snk_spin_weights_r2_2, snk_weights_r2_2, hex_src_psi_re, hex_src_psi_im, hex_snk_psi_re, hex_snk_psi_im);
   make_hex_correlator(H_r2_re, H_r2_im, B1_prop_re, B1_prop_im, B2_prop_re, B2_prop_im, perms, sigs, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, overall_weight/sqrt(2.0), snk_color_weights_r2_2, snk_spin_weights_r2_2, snk_weights_r2_2, hex_src_psi_re, hex_src_psi_im, hex_snk_psi_re, hex_snk_psi_im);
   make_hex_correlator(H_r3_re, H_r3_im, B1_prop_re, B1_prop_im, B2_prop_re, B2_prop_im, perms, sigs, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, overall_weight, snk_color_weights_r3, snk_spin_weights_r3, snk_weights_r3, hex_src_psi_re, hex_src_psi_im, hex_snk_psi_re, hex_snk_psi_im);
   for (m=0; m<NsrcHex; m++) {
      for (n=0; n<NsnkHex; n++) {
         for (t=0; t<Nt; t++) {
            C_re[correlator_index(0,Nsrc+m,Nsnk+n,t)] = H_0_re[one_hex_correlator_index(m,n,t)];
            C_im[correlator_index(0,Nsrc+m,Nsnk+n,t)] = H_0_im[one_hex_correlator_index(m,n,t)];
            C_re[correlator_index(1,Nsrc+m,Nsnk+n,t)] = H_r1_re[one_hex_correlator_index(m,n,t)];
            C_im[correlator_index(1,Nsrc+m,Nsnk+n,t)] = H_r1_im[one_hex_correlator_index(m,n,t)];
            C_re[correlator_index(2,Nsrc+m,Nsnk+n,t)] = H_r2_re[one_hex_correlator_index(m,n,t)];
            C_im[correlator_index(2,Nsrc+m,Nsnk+n,t)] = H_r2_im[one_hex_correlator_index(m,n,t)];
            C_re[correlator_index(3,Nsrc+m,Nsnk+n,t)] = H_r3_re[one_hex_correlator_index(m,n,t)];
            C_im[correlator_index(3,Nsrc+m,Nsnk+n,t)] = H_r3_im[one_hex_correlator_index(m,n,t)];
         }
      }
   }
   /* BB_H */
   make_dibaryon_hex_correlator(BB_H_0_re, BB_H_0_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B2_Blocal_r2_re, B2_Blocal_r2_im, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_1, snk_spin_weights_1, snk_weights_1, hex_snk_psi_re, hex_snk_psi_im);
   make_dibaryon_hex_correlator(BB_H_0_re, BB_H_0_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B2_Blocal_r2_re, B2_Blocal_r2_im, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_2, snk_spin_weights_2, snk_weights_2, hex_snk_psi_re, hex_snk_psi_im);
   make_dibaryon_hex_correlator(BB_H_0_re, BB_H_0_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B2_Blocal_r1_re, B2_Blocal_r1_im, perms, sigs, -1.0*overall_weight/sqrt(2.0), snk_color_weights_1, snk_spin_weights_1, snk_weights_1, hex_snk_psi_re, hex_snk_psi_im);
   make_dibaryon_hex_correlator(BB_H_0_re, BB_H_0_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B2_Blocal_r1_re, B2_Blocal_r1_im, perms, sigs, -1.0*overall_weight/sqrt(2.0), snk_color_weights_2, snk_spin_weights_2, snk_weights_2, hex_snk_psi_re, hex_snk_psi_im);
   make_dibaryon_hex_correlator(BB_H_r1_re, BB_H_r1_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B2_Blocal_r1_re, B2_Blocal_r1_im, perms, sigs, overall_weight, snk_color_weights_r1, snk_spin_weights_r1, snk_weights_r1, hex_snk_psi_re, hex_snk_psi_im);
   make_dibaryon_hex_correlator(BB_H_r2_re, BB_H_r2_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B2_Blocal_r2_re, B2_Blocal_r2_im, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_r2_1, snk_spin_weights_r2_1, snk_weights_r2_1, hex_snk_psi_re, hex_snk_psi_im);
   make_dibaryon_hex_correlator(BB_H_r2_re, BB_H_r2_im, B1_Blocal_r1_re, B1_Blocal_r1_im, B2_Blocal_r2_re, B2_Blocal_r2_im, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_r2_2, snk_spin_weights_r2_2, snk_weights_r2_2, hex_snk_psi_re, hex_snk_psi_im);
   make_dibaryon_hex_correlator(BB_H_r2_re, BB_H_r2_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B2_Blocal_r1_re, B2_Blocal_r1_im, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_r2_1, snk_spin_weights_r2_1, snk_weights_r2_1, hex_snk_psi_re, hex_snk_psi_im);
   make_dibaryon_hex_correlator(BB_H_r2_re, BB_H_r2_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B2_Blocal_r1_re, B2_Blocal_r1_im, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_r2_2, snk_spin_weights_r2_2, snk_weights_r2_2, hex_snk_psi_re, hex_snk_psi_im);
   make_dibaryon_hex_correlator(BB_H_r3_re, BB_H_r3_im, B1_Blocal_r2_re, B1_Blocal_r2_im, B2_Blocal_r2_re, B2_Blocal_r2_im, perms, sigs, overall_weight, snk_color_weights_r3, snk_spin_weights_r3, snk_weights_r3, hex_snk_psi_re, hex_snk_psi_im);
   for (m=0; m<Nsrc; m++) {
      for (n=0; n<NsnkHex; n++) {
         for (t=0; t<Nt; t++) {
            C_re[correlator_index(0,m,Nsnk+n,t)] = BB_H_0_re[one_hex_correlator_index(m,n,t)];
            C_im[correlator_index(0,m,Nsnk+n,t)] = BB_H_0_im[one_hex_correlator_index(m,n,t)];
            C_re[correlator_index(1,m,Nsnk+n,t)] = BB_H_r1_re[one_hex_correlator_index(m,n,t)];
            C_im[correlator_index(1,m,Nsnk+n,t)] = BB_H_r1_im[one_hex_correlator_index(m,n,t)];
            C_re[correlator_index(2,m,Nsnk+n,t)] = BB_H_r2_re[one_hex_correlator_index(m,n,t)];
            C_im[correlator_index(2,m,Nsnk+n,t)] = BB_H_r2_im[one_hex_correlator_index(m,n,t)];
            C_re[correlator_index(3,m,Nsnk+n,t)] = BB_H_r3_re[one_hex_correlator_index(m,n,t)];
            C_im[correlator_index(3,m,Nsnk+n,t)] = BB_H_r3_im[one_hex_correlator_index(m,n,t)];
         }
      }
   } 
   /* H_BB */
   if (snk_entangled == 0) {
      double* snk_B1_Blocal_r1_re = malloc(Nt * Nc * Ns * Nc * Ns * Vsrc * Nc * Ns * Nsnk * sizeof (double));
      double* snk_B1_Blocal_r1_im = malloc(Nt * Nc * Ns * Nc * Ns * Vsrc * Nc * Ns * Nsnk * sizeof (double));
      double* snk_B1_Blocal_r2_re = malloc(Nt * Nc * Ns * Nc * Ns * Vsrc * Nc * Ns * Nsnk * sizeof (double));
      double* snk_B1_Blocal_r2_im = malloc(Nt * Nc * Ns * Nc * Ns * Vsrc * Nc * Ns * Nsnk * sizeof (double));
      double* snk_B2_Blocal_r1_re = malloc(Nt * Nc * Ns * Nc * Ns * Vsrc * Nc * Ns * Nsnk * sizeof (double));
      double* snk_B2_Blocal_r1_im = malloc(Nt * Nc * Ns * Nc * Ns * Vsrc * Nc * Ns * Nsnk * sizeof (double));
      double* snk_B2_Blocal_r2_re = malloc(Nt * Nc * Ns * Nc * Ns * Vsrc * Nc * Ns * Nsnk * sizeof (double));
      double* snk_B2_Blocal_r2_im = malloc(Nt * Nc * Ns * Nc * Ns * Vsrc * Nc * Ns * Nsnk * sizeof (double));
      make_local_snk_block(snk_B1_Blocal_r1_re, snk_B1_Blocal_r1_im, B1_prop_re, B1_prop_im, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, snk_psi_B1_re, snk_psi_B1_im);
      make_local_snk_block(snk_B1_Blocal_r2_re, snk_B1_Blocal_r2_im, B1_prop_re, B1_prop_im, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, snk_psi_B1_re, snk_psi_B1_im);
      make_local_snk_block(snk_B2_Blocal_r1_re, snk_B2_Blocal_r1_im, B2_prop_re, B2_prop_im, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, snk_psi_B2_re, snk_psi_B2_im);
      make_local_snk_block(snk_B2_Blocal_r2_re, snk_B2_Blocal_r2_im, B2_prop_re, B2_prop_im, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, snk_psi_B2_re, snk_psi_B2_im);
      make_hex_dibaryon_correlator(H_BB_0_re, H_BB_0_im, snk_B1_Blocal_r1_re, snk_B1_Blocal_r1_im, snk_B2_Blocal_r2_re, snk_B2_Blocal_r2_im, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_1, snk_spin_weights_1, snk_weights_1, hex_src_psi_re, hex_src_psi_im);
      make_hex_dibaryon_correlator(H_BB_0_re, H_BB_0_im, snk_B1_Blocal_r1_re, snk_B1_Blocal_r1_im, snk_B2_Blocal_r2_re, snk_B2_Blocal_r2_im, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_2, snk_spin_weights_2, snk_weights_2, hex_src_psi_re, hex_src_psi_im);
      make_hex_dibaryon_correlator(H_BB_0_re, H_BB_0_im, snk_B1_Blocal_r2_re, snk_B1_Blocal_r2_im, snk_B2_Blocal_r1_re, snk_B2_Blocal_r1_im, perms, sigs, -1.0*overall_weight/sqrt(2.0), snk_color_weights_1, snk_spin_weights_1, snk_weights_1, hex_src_psi_re, hex_src_psi_im);
      make_hex_dibaryon_correlator(H_BB_0_re, H_BB_0_im, snk_B1_Blocal_r2_re, snk_B1_Blocal_r2_im, snk_B2_Blocal_r1_re, snk_B2_Blocal_r1_im, perms, sigs, -1.0*overall_weight/sqrt(2.0), snk_color_weights_2, snk_spin_weights_2, snk_weights_2, hex_src_psi_re, hex_src_psi_im);
      make_hex_dibaryon_correlator(H_BB_r1_re, H_BB_r1_im, snk_B1_Blocal_r1_re, snk_B1_Blocal_r1_im, snk_B2_Blocal_r1_re, snk_B2_Blocal_r1_im, perms, sigs, overall_weight, snk_color_weights_r1, snk_spin_weights_r1, snk_weights_r1, hex_src_psi_re, hex_src_psi_im);
      make_hex_dibaryon_correlator(H_BB_r2_re, H_BB_r2_im, snk_B1_Blocal_r1_re, snk_B1_Blocal_r1_im, snk_B2_Blocal_r2_re, snk_B2_Blocal_r2_im, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_r2_1, snk_spin_weights_r2_1, snk_weights_r2_1, hex_src_psi_re, hex_src_psi_im);
      make_hex_dibaryon_correlator(H_BB_r2_re, H_BB_r2_im, snk_B1_Blocal_r1_re, snk_B1_Blocal_r1_im, snk_B2_Blocal_r2_re, snk_B2_Blocal_r2_im, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_r2_2, snk_spin_weights_r2_2, snk_weights_r2_2, hex_src_psi_re, hex_src_psi_im);
      make_hex_dibaryon_correlator(H_BB_r2_re, H_BB_r2_im, snk_B1_Blocal_r2_re, snk_B1_Blocal_r2_im, snk_B2_Blocal_r1_re, snk_B2_Blocal_r1_im, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_r2_1, snk_spin_weights_r2_1, snk_weights_r2_1, hex_src_psi_re, hex_src_psi_im);
      make_hex_dibaryon_correlator(H_BB_r2_re, H_BB_r2_im, snk_B1_Blocal_r2_re, snk_B1_Blocal_r2_im, snk_B2_Blocal_r1_re, snk_B2_Blocal_r1_im, perms, sigs, overall_weight/sqrt(2.0), snk_color_weights_r2_2, snk_spin_weights_r2_2, snk_weights_r2_2, hex_src_psi_re, hex_src_psi_im);
      make_hex_dibaryon_correlator(H_BB_r3_re, H_BB_r3_im, snk_B1_Blocal_r2_re, snk_B1_Blocal_r2_im, snk_B2_Blocal_r2_re, snk_B2_Blocal_r2_im, perms, sigs, overall_weight, snk_color_weights_r3, snk_spin_weights_r3, snk_weights_r3, hex_src_psi_re, hex_src_psi_im);
      for (m=0; m<NsrcHex; m++) {
         for (n=0; n<Nsnk; n++) {
            for (t=0; t<Nt; t++) {
               C_re[correlator_index(0,Nsrc+m,n,t)] = H_BB_0_re[one_correlator_index(m,n,t)];
               C_im[correlator_index(0,Nsrc+m,n,t)] = H_BB_0_im[one_correlator_index(m,n,t)];
               C_re[correlator_index(1,Nsrc+m,n,t)] = H_BB_r1_re[one_correlator_index(m,n,t)];
               C_im[correlator_index(1,Nsrc+m,n,t)] = H_BB_r1_im[one_correlator_index(m,n,t)];
               C_re[correlator_index(2,Nsrc+m,n,t)] = H_BB_r2_re[one_correlator_index(m,n,t)];
               C_im[correlator_index(2,Nsrc+m,n,t)] = H_BB_r2_im[one_correlator_index(m,n,t)];
               C_re[correlator_index(3,Nsrc+m,n,t)] = H_BB_r3_re[one_correlator_index(m,n,t)];
               C_im[correlator_index(3,Nsrc+m,n,t)] = H_BB_r3_im[one_correlator_index(m,n,t)];
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
   }
   else {
      for (m=0; m<NsrcHex; m++) {
         for (n=0; n<Nsnk; n++) {
            for (t=0; t<Nt; t++) {
               C_re[correlator_index(0,Nsrc+m,n,t)] = BB_H_0_re[one_correlator_index(m,n,t)];
               C_im[correlator_index(0,Nsrc+m,n,t)] = -BB_H_0_im[one_correlator_index(m,n,t)];
               C_re[correlator_index(1,Nsrc+m,n,t)] = BB_H_r1_re[one_correlator_index(m,n,t)];
               C_im[correlator_index(1,Nsrc+m,n,t)] = -BB_H_r1_im[one_correlator_index(m,n,t)];
               C_re[correlator_index(2,Nsrc+m,n,t)] = BB_H_r2_re[one_correlator_index(m,n,t)];
               C_im[correlator_index(2,Nsrc+m,n,t)] = -BB_H_r2_im[one_correlator_index(m,n,t)];
               C_re[correlator_index(3,Nsrc+m,n,t)] = BB_H_r3_re[one_correlator_index(m,n,t)];
               C_im[correlator_index(3,Nsrc+m,n,t)] = -BB_H_r3_im[one_correlator_index(m,n,t)];
            }
         }
      }
   } 
   /* clean up */
   free(B1_Blocal_r1_re);
   free(B1_Blocal_r1_im);
   free(B1_Blocal_r2_re);
   free(B1_Blocal_r2_im);
   free(B1_Bsingle_r1_re);
   free(B1_Bsingle_r1_im);
   free(B1_Bsingle_r2_re);
   free(B1_Bsingle_r2_im);
   free(B1_Bdouble_r1_re);
   free(B1_Bdouble_r1_im);
   free(B1_Bdouble_r2_re);
   free(B1_Bdouble_r2_im);
   free(B2_Blocal_r1_re);
   free(B2_Blocal_r1_im);
   free(B2_Blocal_r2_re);
   free(B2_Blocal_r2_im);
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
   free(H_0_re);
   free(H_0_im);
   free(H_r1_re);
   free(H_r1_im);
   free(H_r2_re);
   free(H_r2_im);
   free(H_r3_re);
   free(H_r3_im);
   free(H_BB_0_re);
   free(H_BB_0_im);
   free(H_BB_r1_re);
   free(H_BB_r1_im);
   free(H_BB_r2_re);
   free(H_BB_r2_im);
   free(H_BB_r3_re);
   free(H_BB_r3_im);
   free(BB_H_0_re);
   free(BB_H_0_im);
   free(BB_H_r1_re);
   free(BB_H_r1_im);
   free(BB_H_r2_re);
   free(BB_H_r2_im);
   free(BB_H_r3_re);
   free(BB_H_r3_im);
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
    const double* snk_psi_im) {
   /* indices */
   int n, m, t, x, iC, iS, jC, jS, kC, kS, wnum;
   /* create block */
   double* Blocal_r1_re = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Nsrc * sizeof (double));
   double* Blocal_r1_im = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Nsrc * sizeof (double));
   double* Blocal_r2_re = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Nsrc * sizeof (double));
   double* Blocal_r2_im = malloc(Nt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Nsrc * sizeof (double));
   make_local_block(Blocal_r1_re, Blocal_r1_im, prop_re, prop_im, src_color_weights_r1, src_spin_weights_r1, src_weights_r1, src_psi_re, src_psi_im);
   make_local_block(Blocal_r2_re, Blocal_r2_im, prop_re, prop_im, src_color_weights_r2, src_spin_weights_r2, src_weights_r2, src_psi_re, src_psi_im);
   /* create baryon correlator */

   for (wnum=0; wnum<Nw; wnum++) {
      iC = src_color_weights_r1[src_weight_index(wnum,0)];
      iS = src_spin_weights_r1[src_weight_index(wnum,0)];
      jC = src_color_weights_r1[src_weight_index(wnum,1)];
      jS = src_spin_weights_r1[src_weight_index(wnum,1)];
      kC = src_color_weights_r1[src_weight_index(wnum,2)];
      kS = src_spin_weights_r1[src_weight_index(wnum,2)];
      for (m=0; m<Nsrc; m++) {
         for (n=0; n<Nsnk; n++) {
            for (t=0; t<Nt; t++) {
               for (x=0; x<Vsnk; x++) {
                  C_re[B1_correlator_index(0,m,n,t)] += src_weights_r1[wnum] * (snk_psi_re[snk_one_psi_index(x,n)] * Blocal_r1_re[Blocal_index(t,iC,iS,kC,kS,x,jC,jS,m)] - snk_psi_im[snk_one_psi_index(x,n)] * Blocal_r1_im[Blocal_index(t,iC,iS,kC,kS,x,jC,jS,m)]);
                  C_im[B1_correlator_index(0,m,n,t)] += src_weights_r1[wnum] * (snk_psi_re[snk_one_psi_index(x,n)] * Blocal_r1_im[Blocal_index(t,iC,iS,kC,kS,x,jC,jS,m)] - snk_psi_im[snk_one_psi_index(x,n)] * Blocal_r1_re[Blocal_index(t,iC,iS,kC,kS,x,jC,jS,m)]);
               }
            }
         }
      }
   }
   for (wnum=0; wnum<Nw; wnum++) {
      iC = src_color_weights_r2[src_weight_index(wnum,0)];
      iS = src_spin_weights_r2[src_weight_index(wnum,0)];
      jC = src_color_weights_r2[src_weight_index(wnum,1)];
      jS = src_spin_weights_r2[src_weight_index(wnum,1)];
      kC = src_color_weights_r2[src_weight_index(wnum,2)];
      kS = src_spin_weights_r2[src_weight_index(wnum,2)];
      for (m=0; m<Nsrc; m++) {
         for (n=0; n<Nsnk; n++) {
            for (t=0; t<Nt; t++) {
               for (x=0; x<Vsnk; x++) {
                  C_re[B1_correlator_index(1,m,n,t)] += src_weights_r2[wnum] * (snk_psi_re[snk_one_psi_index(x,n)] * Blocal_r2_re[Blocal_index(t,iC,iS,kC,kS,x,jC,jS,m)] - snk_psi_im[snk_one_psi_index(x,n)] * Blocal_r2_im[Blocal_index(t,iC,iS,kC,kS,x,jC,jS,m)]);
                  C_im[B1_correlator_index(1,m,n,t)] += src_weights_r2[wnum] * (snk_psi_re[snk_one_psi_index(x,n)] * Blocal_r2_im[Blocal_index(t,iC,iS,kC,kS,x,jC,jS,m)] - snk_psi_im[snk_one_psi_index(x,n)] * Blocal_r2_re[Blocal_index(t,iC,iS,kC,kS,x,jC,jS,m)]);
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
