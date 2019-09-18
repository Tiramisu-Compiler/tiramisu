#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>
#include <stdio.h>
#include <complex.h>
#include <time.h>
#include <stdbool.h>
#include <math.h>

void make_local_block(std::complex<double> Blocal[Nc][Ns][Nc][Ns][Nc][Ns][Nsrc], 
   const std::complex<double> prop[Nq][Nc][Ns][Nc][Ns][Vsrc], 
   const int color_weights[Nw][Nq], 
   const int spin_weights[Nw][Nq], 
   const double weights[Nw],
   const std::complex<double> psi[Vsrc][Nsrc]) {
   /* indices */
   int iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, iC, iS, jC, jS, kC, kS, x, t, y, wnum, n;
   time_t start,end;
   time (&start);
   double dif;
   /* initialize */
   for (iCprime=0; iCprime<Nc; iCprime++) {
      for (iSprime=0; iSprime<Ns; iSprime++) {
         for (jCprime=0; jCprime<Nc; jCprime++) {
            for (jSprime=0; jSprime<Ns; jSprime++) {
               for (kCprime=0; kCprime<Nc; kCprime++) {
                  for (kSprime=0; kSprime<Ns; kSprime++) {
			    for (n=0; n<Nsrc; n++) {
                              Blocal[iCprime][iSprime][jCprime][jSprime][kCprime][kSprime][n] = 0.0;
                           }
                  }
               }
            }
         }
      }
   }
   /* build local (no quark exchange) block */
   for (wnum=0; wnum<Nw; wnum++) {
      int iC = color_weights[wnum][0];
      int iS = spin_weights[wnum][0];
      int jC = color_weights[wnum][1];
      int jS = spin_weights[wnum][1];
      int kC = color_weights[wnum][2];
      int kS = spin_weights[wnum][2];
      //printf("weight %d %d %d %d %d %d %4.9f \n", iC, iS, jC, jS, kC, kS, weights[wnum]);
      for (iCprime=0; iCprime<Nc; iCprime++) {
         for (iSprime=0; iSprime<Ns; iSprime++) {
            for (jCprime=0; jCprime<Nc; jCprime++) {
               for (jSprime=0; jSprime<Ns; jSprime++) {
                  for (kCprime=0; kCprime<Nc; kCprime++) {
                     for (kSprime=0; kSprime<Ns; kSprime++) {
                              for (n=0; n<Nsrc; n++) {
                                 for (y=0; y<Vsrc; y++) {
                                    Blocal[iCprime][iSprime][jCprime][jSprime][kCprime][kSprime][n] += weights[wnum] * psi[y][n] * ( prop[0][iCprime][iSprime][iC][iS][y] * prop[2][kCprime][kSprime][kC][kS][y] - prop[0][kCprime][kSprime][iC][iS][y] * prop[2][iCprime][iSprime][kC][kS][y] ) * prop[1][jCprime][jSprime][jC][jS][y];
                                 }
                              }
                     }
                  }
               }
            }
         }
      }
   }
   time (&end);
   dif = difftime (end,start);
   //printf("built local baryon block in seconds %5.3f\n",dif);
}

void make_single_block(std::complex<double> Bsingle[Nc][Ns][Nc][Ns][Nc][Ns][Nsrc], 
   const std::complex<double> prop_1[Nq][Nc][Ns][Nc][Ns][Vsrc], 
   const std::complex<double> prop_2[Nq][Nc][Ns][Nc][Ns][Vsrc], 
   const int color_weights[Nw][Nq], 
   const int spin_weights[Nw][Nq], 
   const double weights[Nw],
   const std::complex<double> psi[Vsrc][Nsrc],
   std::complex<double> Q[Nc][Ns][Nc][Ns][Nc][Ns][Vsrc]) {
   /* indices */
   int iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, iC, iS, jC, jS, kC, kS, x, x1, x2, t, y, wnum, n; 
   time_t start,end;
   time (&start);
   double dif;
   /* initialize */
   for (iCprime=0; iCprime<Nc; iCprime++) {
      for (iSprime=0; iSprime<Ns; iSprime++) {
         for (kCprime=0; kCprime<Nc; kCprime++) {
            for (kSprime=0; kSprime<Ns; kSprime++) {
               for (jC=0; jC<Nc; jC++) {
                  for (jS=0; jS<Ns; jS++) {
                           for (y=0; y<Vsrc; y++) {
                                 Q[iCprime][iSprime][kCprime][kSprime][jC][jS][y] = 0.0;
                           }
                  }
               }
            }
         }
      }
   }
   for (iCprime=0; iCprime<Nc; iCprime++) {
      for (iSprime=0; iSprime<Ns; iSprime++) {
         for (jCprime=0; jCprime<Nc; jCprime++) {
            for (jSprime=0; jSprime<Ns; jSprime++) {
               for (kCprime=0; kCprime<Nc; kCprime++) {
                  for (kSprime=0; kSprime<Ns; kSprime++) {
                              for (n=0; n<Nsrc; n++) {
                                 Bsingle[iCprime][iSprime][jCprime][jSprime][kCprime][kSprime][n] = 0.0;
                              }
                  }
               }
            }
         }
      }
   }
   /* build diquarks */
   for (wnum=0; wnum<Nw; wnum++) {
      int iC = color_weights[wnum][0];
      int iS = spin_weights[wnum][0];
      int jC = color_weights[wnum][1];
      int jS = spin_weights[wnum][1];
      int kC = color_weights[wnum][2];
      int kS = spin_weights[wnum][2];
      for (iCprime=0; iCprime<Nc; iCprime++) {
         for (iSprime=0; iSprime<Ns; iSprime++) {
            for (kCprime=0; kCprime<Nc; kCprime++) {
               for (kSprime=0; kSprime<Ns; kSprime++) {
                        for (y=0; y<Vsrc; y++) {
                              Q[iCprime][iSprime][kCprime][kSprime][jC][jS][y] += weights[wnum]  * ( prop_1[0][iCprime][iSprime][iC][iS][y] * prop_1[2][kCprime][kSprime][kC][kS][y] - prop_1[0][kCprime][kSprime][iC][iS][y] * prop_1[2][iCprime][iSprime][kC][kS][y] );
                        }
               }
            }
         }
      }
   }
   /* build q2-exchange block */
   for (iCprime=0; iCprime<Nc; iCprime++) {
      for (iSprime=0; iSprime<Ns; iSprime++) {
         for (jCprime=0; jCprime<Nc; jCprime++) {
            for (jSprime=0; jSprime<Ns; jSprime++) {
               for (kCprime=0; kCprime<Nc; kCprime++) {
                  for (kSprime=0; kSprime<Ns; kSprime++) {
                              for (y=0; y<Vsrc; y++) {
                                 for (jC=0; jC<Nc; jC++) {
                                   for (jS=0; jS<Ns; jS++) {
                                     Bsingle[iCprime][iSprime][jCprime][jSprime][kCprime][kSprime][n] += psi[y][n] * Q[iCprime][iSprime][kCprime][kSprime][jC][jS][y] * prop_2[1][jCprime][jSprime][jC][jS][y];
                                    }
                                 }
                              }
                  }
               }
            }
         }
      }
   }
   time (&end);
   dif = difftime(end,start);
   // printf("built single exchange baryon block in seconds %5.3f\n",dif);
}

void make_double_block(std::complex<double> Bdouble[Nc][Ns][Nc][Ns][Nc][Ns][Nsrc], 
    const std::complex<double> prop_1[Nq][Nc][Ns][Nc][Ns][Vsrc], 
    const std::complex<double> prop_2[Nq][Nc][Ns][Nc][Ns][Vsrc], 
    const int color_weights[Nw][Nq], 
    const int spin_weights[Nw][Nq], 
    const double weights[Nw],
    const std::complex<double> psi[Vsrc][Nsrc],
    std::complex<double> O[Nc][Ns][Nc][Ns][Nc][Ns][Vsrc],
    std::complex<double> P[Nc][Ns][Nc][Ns][Nc][Ns][Vsrc]) {
   /* indices */
   int iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, iC, iS, jC, jS, kC, kS, x, x1, x2, t, y, wnum, n;
   time_t start,end;
   time (&start);
   double dif;
   /* initialize */
   for (jCprime=0; jCprime<Nc; jCprime++) {
      for (jSprime=0; jSprime<Ns; jSprime++) {
         for (kCprime=0; kCprime<Nc; kCprime++) {
            for (kSprime=0; kSprime<Ns; kSprime++) {
               for (jC=0; jC<Nc; jC++) {
                  for (jS=0; jS<Ns; jS++) {
                           for (y=0; y<Vsrc; y++) {
                                 O[jCprime][jSprime][kCprime][kSprime][jC][jS][y] = 0.0;
                                 P[jCprime][jSprime][kCprime][kSprime][jC][jS][y] = 0.0;
                           }
                  }
               }
            }
         }
      }
   }
   for (iCprime=0; iCprime<Nc; iCprime++) {
      for (iSprime=0; iSprime<Ns; iSprime++) {
         for (jCprime=0; jCprime<Nc; jCprime++) {
            for (jSprime=0; jSprime<Ns; jSprime++) {
               for (kCprime=0; kCprime<Nc; kCprime++) {
                  for (kSprime=0; kSprime<Ns; kSprime++) {
                              for (n=0; n<Nsrc; n++) {
                                 Bdouble[iCprime][iSprime][jCprime][jSprime][kCprime][kSprime][n] = 0.0;
                              }
                  }
               }
            }
         }
      }
   }
   /* build diquarks */
   for (wnum=0; wnum<Nw; wnum++) {
      int iC = color_weights[wnum][0];
      int iS = spin_weights[wnum][0];
      int jC = color_weights[wnum][1];
      int jS = spin_weights[wnum][1];
      int kC = color_weights[wnum][2];
      int kS = spin_weights[wnum][2];
      for (jCprime=0; jCprime<Nc; jCprime++) {
         for (jSprime=0; jSprime<Ns; jSprime++) {
            for (kCprime=0; kCprime<Nc; kCprime++) {
               for (kSprime=0; kSprime<Ns; kSprime++) {
                        for (y=0; y<Vsrc; y++) {
                              O[jCprime][jSprime][kCprime][kSprime][iC][iS][y] += weights[wnum] * prop_1[1][jCprime][jSprime][jC][jS][y] * prop_1[2][kCprime][kSprime][kC][kS][y];
                              P[jCprime][jSprime][kCprime][kSprime][kC][kS][y] += weights[wnum] * prop_1[0][kCprime][kSprime][iC][iS][y] * prop_1[1][jCprime][jSprime][jC][jS][y];
                        }
               }
            }
         }
      }
   }
   /* build q1/3-exchange block */
   for (iCprime=0; iCprime<Nc; iCprime++) {
      for (iSprime=0; iSprime<Ns; iSprime++) {
         for (jCprime=0; jCprime<Nc; jCprime++) {
            for (jSprime=0; jSprime<Ns; jSprime++) {
               for (kCprime=0; kCprime<Nc; kCprime++) {
                  for (kSprime=0; kSprime<Ns; kSprime++) {
                     for (x1=0; x1<Vsnk; x1++) {
                        for (x2=0; x2<Vsnk; x2++) {
                           for (t=0; t<Lt; t++) {
                              for (y=0; y<Vsrc; y++) {
                                 for (iC=0; iC<Nc; iC++) {
                                    for (iS=0; iS<Ns; iS++) {
                                       for (n=0; n<Nsrc; n++) {
                                          Bdouble[iCprime][iSprime][jCprime][jSprime][kCprime][kSprime][n] += psi[y][n] * prop_2[0][iCprime][iSprime][iC][iS][y] * O[jCprime][jSprime][kCprime][kSprime][iC][iS][y];
                                       }
                                    }
                                 }
                                 for (kC=0; kC<Nc; kC++) {
                                    for (kS=0; kS<Ns; kS++) {
                                       for (n=0; n<Nsrc; n++) {
                                          Bdouble[iCprime][iSprime][jCprime][jSprime][kCprime][kSprime][n] -= psi[y][n] *P[jCprime][jSprime][kCprime][kSprime][kC][kS][y] * prop_2[2][iCprime][iSprime][kC][kS][y];
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
   time (&end);
   dif = difftime(end,start);
   //printf("built double exchange baryon block in seconds %5.3f\n",dif);
}

