#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>
#include <stdio.h>
#include <complex.h>
#include <time.h>
#include <stdbool.h>
#include <math.h>

void make_local_block(std::complex<double> Blocal[Nsrc][Nc][Ns][Nc][Ns][Nc][Ns][Vsnk][Lt], 
   const std::complex<double> prop[Nq][Nc][Ns][Nc][Ns][Vsnk][Lt][Vsrc], 
   const int color_weights[Nw][Nq], 
   const int spin_weights[Nw][Nq], 
   const double weights[Nw],
   const std::complex<double> psi[Nsrc][Vsrc]) {
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
                     for (x=0; x<Vsnk; x++) {
                        for (t=0; t<Lt; t++) {
			    for (n=0; n<Nsrc; n++) {
                              Blocal[n][iCprime][iSprime][jCprime][jSprime][kCprime][kSprime][x][t] = 0.0;
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
                        for (x=0; x<Vsnk; x++) {
                           for (t=0; t<Lt; t++) {
                              for (n=0; n<Nsrc; n++) {
                                 for (y=0; y<Vsrc; y++) {
                                    Blocal[n][iCprime][iSprime][jCprime][jSprime][kCprime][kSprime][x][t] += weights[wnum] * psi[n][y] * ( prop[0][iCprime][iSprime][iC][iS][x][t][y] * prop[2][kCprime][kSprime][kC][kS][x][t][y] - prop[0][kCprime][kSprime][iC][iS][x][t][y] * prop[2][iCprime][iSprime][kC][kS][x][t][y] ) * prop[1][jCprime][jSprime][jC][jS][x][t][y];
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
   dif = difftime (end,start);
   //printf("built local baryon block in seconds %5.3f\n",dif);
}

void make_single_block(std::complex<double> Bsingle[Nsrc][Nc][Ns][Nc][Ns][Nc][Ns][Vsnk][Vsnk][Lt], 
   const std::complex<double> prop[Nq][Nc][Ns][Nc][Ns][Vsnk][Lt][Vsrc], 
   const int color_weights[Nw][Nq], 
   const int spin_weights[Nw][Nq], 
   const double weights[Nw],
   const std::complex<double> psi[Nsrc][Vsrc],
   std::complex<double> Q[Nsrc][Nc][Ns][Nc][Ns][Nc][Ns][Vsnk][Lt][Vsrc]) {
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
                     for (x=0; x<Vsnk; x++) {
                        for (t=0; t<Lt; t++) {
                           for (y=0; y<Vsrc; y++) {
                              for (n=0; n<Nsrc; n++) {
                                 Q[n][iCprime][iSprime][kCprime][kSprime][jC][jS][x][t][y] = 0.0;
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
   for (iCprime=0; iCprime<Nc; iCprime++) {
      for (iSprime=0; iSprime<Ns; iSprime++) {
         for (jCprime=0; jCprime<Nc; jCprime++) {
            for (jSprime=0; jSprime<Ns; jSprime++) {
               for (kCprime=0; kCprime<Nc; kCprime++) {
                  for (kSprime=0; kSprime<Ns; kSprime++) {
                     for (x1=0; x1<Vsnk; x1++) {
                        for (x2=0; x2<Vsnk; x2++) {
                           for (t=0; t<Lt; t++) {
                              for (n=0; n<Nsrc; n++) {
                                 Bsingle[n][iCprime][iSprime][jCprime][jSprime][kCprime][kSprime][x1][x2][t] = 0.0;
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
                  for (x=0; x<Vsnk; x++) {
                     for (t=0; t<Lt; t++) {
                        for (y=0; y<Vsrc; y++) {
                           for (n=0; n<Nsrc; n++) {
                              Q[n][iCprime][iSprime][kCprime][kSprime][jC][jS][x][t][y] += weights[wnum] * psi[n][y] * ( prop[0][iCprime][iSprime][iC][iS][x][t][y] * prop[2][kCprime][kSprime][kC][kS][x][t][y] - prop[0][kCprime][kSprime][iC][iS][x][t][y] * prop[2][iCprime][iSprime][kC][kS][x][t][y] );
                           }
                        }
                     }
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
                     for (x1=0; x1<Vsnk; x1++) {
                        for (x2=0; x2<Vsnk; x2++) {
                           for (t=0; t<Lt; t++) {
                              for (y=0; y<Vsrc; y++) {
                                 for (jC=0; jC<Nc; jC++) {
                                   for (jS=0; jS<Ns; jS++) {
                                    for (n=0; n<Nsrc; n++) {
                                     Bsingle[n][iCprime][iSprime][jCprime][jSprime][kCprime][kSprime][x1][x2][t] += Q[n][iCprime][iSprime][kCprime][kSprime][jC][jS][x1][t][y] * prop[1][jCprime][jSprime][jC][jS][x2][t][y];
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
   // printf("built single exchange baryon block in seconds %5.3f\n",dif);
}

void make_double_block(std::complex<double> Bdouble[Nsrc][Nc][Ns][Nc][Ns][Nc][Ns][Vsnk][Vsnk][Lt], 
    const std::complex<double> prop[Nq][Nc][Ns][Nc][Ns][Vsnk][Lt][Vsrc], 
    const int color_weights[Nw][Nq], 
    const int spin_weights[Nw][Nq], 
    const double weights[Nw],
    const std::complex<double> psi[Nsrc][Vsrc],
    std::complex<double> O[Nsrc][Nc][Ns][Nc][Ns][Nc][Ns][Vsnk][Lt][Vsrc],
    std::complex<double> P[Nsrc][Nc][Ns][Nc][Ns][Nc][Ns][Vsnk][Lt][Vsrc]) {
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
                     for (x=0; x<Vsnk; x++) {
                        for (t=0; t<Lt; t++) {
                           for (y=0; y<Vsrc; y++) {
                              for (n=0; n<Nsrc; n++) {
                                 O[n][jCprime][jSprime][kCprime][kSprime][jC][jS][x][t][y] = 0.0;
                                 P[n][jCprime][jSprime][kCprime][kSprime][jC][jS][x][t][y] = 0.0;
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
   for (iCprime=0; iCprime<Nc; iCprime++) {
      for (iSprime=0; iSprime<Ns; iSprime++) {
         for (jCprime=0; jCprime<Nc; jCprime++) {
            for (jSprime=0; jSprime<Ns; jSprime++) {
               for (kCprime=0; kCprime<Nc; kCprime++) {
                  for (kSprime=0; kSprime<Ns; kSprime++) {
                     for (x1=0; x1<Vsnk; x1++) {
                        for (x2=0; x2<Vsnk; x2++) {
                           for (t=0; t<Lt; t++) {
                              for (n=0; n<Nsrc; n++) {
                                 Bdouble[n][iCprime][iSprime][jCprime][jSprime][kCprime][kSprime][x1][x2][t] = 0.0;
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
                  for (x=0; x<Vsnk; x++) {
                     for (t=0; t<Lt; t++) {
                        for (y=0; y<Vsrc; y++) {
                           for (n=0; n<Nsrc; n++) {
                              O[n][jCprime][jSprime][kCprime][kSprime][iC][iS][x][t][y] += weights[wnum] * psi[n][y] * prop[1][jCprime][jSprime][jC][jS][x][t][y] * prop[2][kCprime][kSprime][kC][kS][x][t][y];
                              P[n][jCprime][jSprime][kCprime][kSprime][kC][kS][x][t][y] += weights[wnum] * psi[n][y] * prop[0][kCprime][kSprime][iC][iS][x][t][y] * prop[1][jCprime][jSprime][jC][jS][x][t][y];
                           }
                        }
                     }
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
                                          Bdouble[n][iCprime][iSprime][jCprime][jSprime][kCprime][kSprime][x1][x2][t] += prop[0][iCprime][iSprime][iC][iS][x2][t][y] * O[n][jCprime][jSprime][kCprime][kSprime][iC][iS][x1][t][y];
                                       }
                                    }
                                 }
                                 for (kC=0; kC<Nc; kC++) {
                                    for (kS=0; kS<Ns; kS++) {
                                       for (n=0; n<Nsrc; n++) {
                                          Bdouble[n][iCprime][iSprime][jCprime][jSprime][kCprime][kSprime][x1][x2][t] -= P[n][jCprime][jSprime][kCprime][kSprime][kC][kS][x1][t][y] * prop[2][iCprime][iSprime][kC][kS][x2][t][y];
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

void make_pion_correlator(std::complex<double> correlator[Lt],
    const std::complex<double> prop[Nq][Nc][Ns][Nc][Ns][Vsnk][Lt][Vsrc]) {
   time_t start,end;
   time (&start);
   double dif;
   /* indices */
   int iCprime, iSprime, iC, iS, x, t, y;
   /* build pion */
   for (iC=0; iC<Nc; iC++) {
      for (iS=0; iS<Ns; iS++) {
         for (iCprime=0; iCprime<Nc; iCprime++) {
            for (iSprime=0; iSprime<Ns; iSprime++) {
               for (x=0; x<Vsnk; x++) {
                  for (y=0; y<Vsrc; y++) {
                     for (t=0; t<Lt; t++) {
                        correlator[t] += prop[0][iCprime][iSprime][iC][iS][x][t][y] * conj(prop[0][iC][iS][iCprime][iSprime][x][t][y]);
                     }
                  }
               }
            }
         }
      }
   }
   time (&end);
   dif = difftime(end,start);
   printf("built pion correlator in seconds %5.3f\n",dif);
}

void make_baryon_correlator(std::complex<double> correlator[Nsrc][Nsnk][Lt],
    const std::complex<double> Blocal[Nsrc][Nc][Ns][Nc][Ns][Nc][Ns][Vsnk][Lt],
    const int color_weights[Nw][Nq], 
    const int spin_weights[Nw][Nq], 
    const double weights[Nw],
    const std::complex<double> psi[Nsnk][Vsnk]) {
   time_t start,end;
   time (&start);
   double dif;
   /* indices */
   int iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, iC, iS, jC, jS, kC, kS, x, t, y, wnum, n, m;
   /* build baryon */
   for (wnum=0; wnum<Nw; wnum++) {
      int iC = color_weights[wnum][0];
      int iS = spin_weights[wnum][0];
      int jC = color_weights[wnum][1];
      int jS = spin_weights[wnum][1];
      int kC = color_weights[wnum][2];
      int kS = spin_weights[wnum][2];
      for (x=0; x<Vsnk; x++) {
         for (t=0; t<Lt; t++) {
            for (n=0; n<Nsrc; n++) {
               for (m=0; m<Nsnk; m++) {
                  correlator[n][m][t] += weights[wnum] * psi[m][x] * Blocal[n][iC][iS][jC][jS][kC][kS][x][t];
               }
            }
         }
      }
   }
   time (&end);
   dif = difftime(end,start);
   printf("built baryon correlator in seconds %5.3f\n",dif);
}


void make_dibaryon_correlator(std::complex<double> correlator[Nsrc][Nsnk][Lt],
    const std::complex<double> Blocal1[Nsrc][Nc][Ns][Nc][Ns][Nc][Ns][Vsnk][Lt],
    const std::complex<double> Bsingle1[Nsrc][Nc][Ns][Nc][Ns][Nc][Ns][Vsnk][Vsnk][Lt], 
    const std::complex<double> Bdouble1[Nsrc][Nc][Ns][Nc][Ns][Nc][Ns][Vsnk][Vsnk][Lt], 
    const std::complex<double> Blocal2[Nsrc][Nc][Ns][Nc][Ns][Nc][Ns][Vsnk][Lt],
    const std::complex<double> Bsingle2[Nsrc][Nc][Ns][Nc][Ns][Nc][Ns][Vsnk][Vsnk][Lt], 
    const std::complex<double> Bdouble2[Nsrc][Nc][Ns][Nc][Ns][Nc][Ns][Vsnk][Vsnk][Lt], 
    const int perms[Nperms][Nq*2], 
    const int sigs[Nperms], 
    const double overall_weight,
    const int color_weights[2][twoNw][Nq], 
    const int spin_weights[2][twoNw][Nq], 
    const double weights[twoNw],
    const std::complex<double> psi2[Nsnk][Vsnk][Vsnk]) 
   {
   time_t start,end;
   time (&start);
   double dif;
   int count = 0;
   int Nb = 2;
   /* indices */
   int iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, iC, iS, jC, jS, kC, kS, x1, x2, t, wnum, nperm, b, n, m;
   bool not_redundant;
   /* build dibaryon */
   for (nperm=0; nperm<Nperms; nperm++) {
      not_redundant = true;
      for (b=0; b<Nb; b++) {
         if (perms[nperm][Nq*b] > perms[nperm][Nq*b+2]) {
            not_redundant = false;
         }
      }
      if (not_redundant) {
         count = count + 1;
         int snk_1_nq[Nb];
         int snk_2_nq[Nb];
         int snk_3_nq[Nb];
         int snk_1[Nb];
         int snk_2[Nb];
         int snk_3[Nb];
         for (b=0; b<Nb; b++) {
            snk_1_nq[b] = perms[nperm][Nq*b] - 1;
            snk_2_nq[b] = perms[nperm][Nq*b+1] - 1;
            snk_3_nq[b] = perms[nperm][Nq*b+2] - 1;
            snk_1[b] = (snk_1_nq[b] - snk_1_nq[b] % Nc) / Nc;
            snk_2[b] = (snk_2_nq[b] - snk_2_nq[b] % Nc) / Nc;
            snk_3[b] = (snk_3_nq[b] - snk_3_nq[b] % Nc) / Nc;
         }
         for (wnum=0; wnum<twoNw; wnum++) {
            int iC1 = color_weights[0][wnum][0];
            int iS1 = spin_weights[0][wnum][0];
            int jC1 = color_weights[0][wnum][1];
            int jS1 = spin_weights[0][wnum][1];
            int kC1 = color_weights[0][wnum][2];
            int kS1 = spin_weights[0][wnum][2];
            int iC2 = color_weights[1][wnum][0];
            int iS2 = spin_weights[1][wnum][0];
            int jC2 = color_weights[1][wnum][1];
            int jS2 = spin_weights[1][wnum][1];
            int kC2 = color_weights[1][wnum][2];
            int kS2 = spin_weights[1][wnum][2];
            for (t=0; t<Lt; t++) {
               for (x1=0; x1<Vsnk; x1++) {
                  for (x2=0; x2<Vsnk; x2++) {
                     for (n=0; n<Nsrc; n++) {
                        for (m=0; m<Nsnk; m++) {
			    std::complex<double> term = sigs[nperm] * overall_weight * weights[wnum] * psi2[m][x1][x2];
                           for (b=0; b<Nb; b++) {
                              if ((snk_1[b] == 0) && (snk_2[b] == 0) && (snk_3[b] == 0)) {
                                 term *= Blocal1[n][iC1][iS1][jC1][jS1][kC1][kS1][x1][t];
                              }
                              else if ((snk_1[b] == 1) && (snk_2[b] == 1) && (snk_3[b] == 1)) {
                                 term *= Blocal2[n][iC2][iS2][jC2][jS2][kC2][kS2][x2][t];
                              }
                              else if ((snk_1[b] == 0) && (snk_3[b] == 0)) {
                                 term *= Bsingle1[n][iC1][iS1][jC1][jS1][kC1][kS1][x1][x2][t];
                              }
                              else if ((snk_1[b] == 1) && (snk_3[b] == 1)) {
                                 term *= Bsingle2[n][iC2][iS2][jC2][jS2][kC2][kS2][x2][x1][t];
                              }
                              else if ((snk_1[b] == 0) && (snk_2[b] == 0)) {
                                 term *= Bdouble1[n][iC1][iS1][jC1][jS1][kC1][kS1][x1][x2][t];
                              }
                              else if ((snk_1[b] == 1) && (snk_2[b] == 1)) {
                                 term *= Bdouble2[n][iC2][iS2][jC2][jS2][kC2][kS2][x2][x1][t];
                              }
                           }
                           correlator[n][m][t] += term;
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
   printf("built dibaryon correlator in seconds %5.3f\n",dif);
}

// Function to optimize
void make_dibaryon_from_props(std::complex<double> C_pi[Lt],
    std::complex<double> C_B1_G1g_r1[Nsrc][Nsnk][Lt],
    std::complex<double> C_B1_G1g_r2[Nsrc][Nsnk][Lt],
    std::complex<double> C_B2_A1g[Nsrc][Nsnk][Lt],
    std::complex<double> C_B2_T1g_r1[Nsrc][Nsnk][Lt],
    std::complex<double> C_B2_T1g_r2[Nsrc][Nsnk][Lt],
    std::complex<double> C_B2_T1g_r3[Nsrc][Nsnk][Lt],
    const std::complex<double> prop[Nq][Nc][Ns][Nc][Ns][Vsnk][Lt][Vsrc], 
    const int B1_G1g_r1_color_weights[Nw][Nq], 
    const int B1_G1g_r1_spin_weights[Nw][Nq], 
    const double B1_G1g_r1_weights[Nw],
    const int B1_G1g_r2_color_weights[Nw][Nq], 
    const int B1_G1g_r2_spin_weights[Nw][Nq], 
    const double B1_G1g_r2_weights[Nw],
    const int perms[Nperms][Nq*2], 
    const int sigs[Nperms], 
    const std::complex<double> psi[Nsrc][Vsrc],
    const std::complex<double> psi1[Nsnk][Vsnk],
    const std::complex<double> psi2[Nsnk][Vsnk][Vsnk]) {
   /* indices */
   int nB1, nB2, nq;
   /* compute dibaryon weights */
   int B2_A1g_spin_weights_1[2][twoNw][Nq];
   int B2_A1g_spin_weights_2[2][twoNw][Nq];
   int B2_T1g_r1_spin_weights[2][twoNw][Nq];
   int B2_T1g_r2_spin_weights_1[2][twoNw][Nq];
   int B2_T1g_r2_spin_weights_2[2][twoNw][Nq];
   int B2_T1g_r3_spin_weights[2][twoNw][Nq];
   int B2_A1g_color_weights_1[2][twoNw][Nq];
   int B2_A1g_color_weights_2[2][twoNw][Nq];
   int B2_T1g_r1_color_weights[2][twoNw][Nq];
   int B2_T1g_r2_color_weights_1[2][twoNw][Nq];
   int B2_T1g_r2_color_weights_2[2][twoNw][Nq];
   int B2_T1g_r3_color_weights[2][twoNw][Nq];
   double B2_A1g_weights_1[twoNw];
   double B2_A1g_weights_2[twoNw];
   double B2_T1g_r1_weights[twoNw];
   double B2_T1g_r2_weights_1[twoNw];
   double B2_T1g_r2_weights_2[twoNw];
   double B2_T1g_r3_weights[twoNw];
   for (nB1=0; nB1<Nw; nB1++) {
      for (nB2=0; nB2<Nw; nB2++) {
         B2_A1g_weights_1[nB1 + nB2*Nw] = 1.0/sqrt(2) * B1_G1g_r1_weights[nB1]*B1_G1g_r2_weights[nB2];
         B2_A1g_weights_2[nB1 + nB2*Nw] = -1.0/sqrt(2) * B1_G1g_r2_weights[nB1]*B1_G1g_r1_weights[nB2];
         B2_T1g_r1_weights[nB1 + nB2*Nw] = B1_G1g_r1_weights[nB1]*B1_G1g_r1_weights[nB2];
         B2_T1g_r2_weights_1[nB1 + nB2*Nw] = 1.0/sqrt(2) * B1_G1g_r1_weights[nB1]*B1_G1g_r2_weights[nB2];
         B2_T1g_r2_weights_2[nB1 + nB2*Nw] = 1.0/sqrt(2) * B1_G1g_r2_weights[nB1]*B1_G1g_r1_weights[nB2];
         B2_T1g_r3_weights[nB1 + nB2*Nw] = B1_G1g_r2_weights[nB1]*B1_G1g_r2_weights[nB2];
         for (nq=0; nq<Nq; nq++) {
            /* A1g */
            B2_A1g_color_weights_1[0][nB1 + nB2*Nw][nq] = B1_G1g_r1_color_weights[nB1][nq];
            B2_A1g_spin_weights_1[0][nB1 + nB2*Nw][nq] = B1_G1g_r1_spin_weights[nB1][nq];
            B2_A1g_color_weights_1[1][nB1 + nB2*Nw][nq] = B1_G1g_r2_color_weights[nB2][nq];
            B2_A1g_spin_weights_1[1][nB1 + nB2*Nw][nq] = B1_G1g_r2_spin_weights[nB2][nq];
            B2_A1g_color_weights_2[0][nB1 + nB2*Nw][nq] = B1_G1g_r2_color_weights[nB1][nq];
            B2_A1g_spin_weights_2[0][nB1 + nB2*Nw][nq] = B1_G1g_r2_spin_weights[nB1][nq];
            B2_A1g_color_weights_2[1][nB1 + nB2*Nw][nq] = B1_G1g_r1_color_weights[nB2][nq];
            B2_A1g_spin_weights_2[1][nB1 + nB2*Nw][nq] = B1_G1g_r1_spin_weights[nB2][nq];
            /* T1g_r1 */
            B2_T1g_r1_color_weights[0][nB1 + nB2*Nw][nq] = B1_G1g_r1_color_weights[nB1][nq];
            B2_T1g_r1_spin_weights[0][nB1 + nB2*Nw][nq] = B1_G1g_r1_spin_weights[nB1][nq];
            B2_T1g_r1_color_weights[1][nB1 + nB2*Nw][nq] = B1_G1g_r1_color_weights[nB2][nq];
            B2_T1g_r1_spin_weights[1][nB1 + nB2*Nw][nq] = B1_G1g_r1_spin_weights[nB2][nq];
            /* T1g_r2 */
            B2_T1g_r2_color_weights_1[0][nB1 + nB2*Nw][nq] = B1_G1g_r1_color_weights[nB1][nq];
            B2_T1g_r2_spin_weights_1[0][nB1 + nB2*Nw][nq] = B1_G1g_r1_spin_weights[nB1][nq];
            B2_T1g_r2_color_weights_1[1][nB1 + nB2*Nw][nq] = B1_G1g_r2_color_weights[nB2][nq];
            B2_T1g_r2_spin_weights_1[1][nB1 + nB2*Nw][nq] = B1_G1g_r2_spin_weights[nB2][nq];
            B2_T1g_r2_color_weights_2[0][nB1 + nB2*Nw][nq] = B1_G1g_r2_color_weights[nB1][nq];
            B2_T1g_r2_spin_weights_2[0][nB1 + nB2*Nw][nq] = B1_G1g_r2_spin_weights[nB1][nq];
            B2_T1g_r2_color_weights_2[1][nB1 + nB2*Nw][nq] = B1_G1g_r1_color_weights[nB2][nq];
            B2_T1g_r2_spin_weights_2[1][nB1 + nB2*Nw][nq] = B1_G1g_r1_spin_weights[nB2][nq];
            /* T1g_r3 */
            B2_T1g_r3_color_weights[0][nB1 + nB2*Nw][nq] = B1_G1g_r2_color_weights[nB1][nq];
            B2_T1g_r3_spin_weights[0][nB1 + nB2*Nw][nq] = B1_G1g_r2_spin_weights[nB1][nq];
            B2_T1g_r3_color_weights[1][nB1 + nB2*Nw][nq] = B1_G1g_r2_color_weights[nB2][nq];
            B2_T1g_r3_spin_weights[1][nB1 + nB2*Nw][nq] = B1_G1g_r2_spin_weights[nB2][nq];
         }
      }
   }
   /* create blocks */
   std::complex<double> G1g_r1_BLocal[Nsrc][Nc][Ns][Nc][Ns][Nc][Ns][Vsnk][Lt];
   std::complex<double> G1g_r1_BSingle[Nsrc][Nc][Ns][Nc][Ns][Nc][Ns][Vsnk][Vsnk][Lt];
   std::complex<double> G1g_r1_BDouble[Nsrc][Nc][Ns][Nc][Ns][Nc][Ns][Vsnk][Vsnk][Lt];
   std::complex<double> G1g_r2_BLocal[Nsrc][Nc][Ns][Nc][Ns][Nc][Ns][Vsnk][Lt];
   std::complex<double> G1g_r2_BSingle[Nsrc][Nc][Ns][Nc][Ns][Nc][Ns][Vsnk][Vsnk][Lt];
   std::complex<double> G1g_r2_BDouble[Nsrc][Nc][Ns][Nc][Ns][Nc][Ns][Vsnk][Vsnk][Lt];
   std::complex<double> Q[Nsrc][Nc][Ns][Nc][Ns][Nc][Ns][Vsnk][Lt][Vsrc];
   std::complex<double> O[Nsrc][Nc][Ns][Nc][Ns][Nc][Ns][Vsnk][Lt][Vsrc];
   std::complex<double> P[Nsrc][Nc][Ns][Nc][Ns][Nc][Ns][Vsnk][Lt][Vsrc];
   make_local_block(G1g_r1_BLocal, prop, B1_G1g_r1_color_weights, B1_G1g_r1_spin_weights, B1_G1g_r1_weights, psi);
   make_single_block(G1g_r1_BSingle, prop, B1_G1g_r1_color_weights, B1_G1g_r1_spin_weights, B1_G1g_r1_weights, psi, Q);
   make_double_block(G1g_r1_BDouble, prop, B1_G1g_r1_color_weights, B1_G1g_r1_spin_weights, B1_G1g_r1_weights, psi, O, P);
   make_local_block(G1g_r2_BLocal, prop, B1_G1g_r2_color_weights, B1_G1g_r2_spin_weights, B1_G1g_r2_weights, psi);
   make_single_block(G1g_r2_BSingle, prop, B1_G1g_r2_color_weights, B1_G1g_r2_spin_weights, B1_G1g_r2_weights, psi, Q);
   make_double_block(G1g_r2_BDouble, prop, B1_G1g_r2_color_weights, B1_G1g_r2_spin_weights, B1_G1g_r2_weights, psi, O, P);
   /* create pion */
   make_pion_correlator(C_pi, prop);
   /* create baryon and test dibaryon blocks */
   make_baryon_correlator(C_B1_G1g_r1, G1g_r1_BLocal, B1_G1g_r1_color_weights, B1_G1g_r1_spin_weights, B1_G1g_r1_weights, psi1);
   make_baryon_correlator(C_B1_G1g_r2, G1g_r2_BLocal, B1_G1g_r2_color_weights, B1_G1g_r2_spin_weights, B1_G1g_r2_weights, psi1);
   /* create dibaryons */
   make_dibaryon_correlator(C_B2_A1g, G1g_r1_BLocal, G1g_r1_BSingle, G1g_r1_BDouble, G1g_r2_BLocal, G1g_r2_BSingle, G1g_r2_BDouble, perms, sigs, 1.0/sqrt(2), B2_A1g_color_weights_1, B2_A1g_spin_weights_1, B2_A1g_weights_1, psi2);
   make_dibaryon_correlator(C_B2_A1g, G1g_r1_BLocal, G1g_r1_BSingle, G1g_r1_BDouble, G1g_r2_BLocal, G1g_r2_BSingle, G1g_r2_BDouble, perms, sigs, 1.0/sqrt(2), B2_A1g_color_weights_2, B2_A1g_spin_weights_2, B2_A1g_weights_2, psi2);
   make_dibaryon_correlator(C_B2_A1g, G1g_r2_BLocal, G1g_r2_BSingle, G1g_r2_BDouble, G1g_r1_BLocal, G1g_r1_BSingle, G1g_r1_BDouble, perms, sigs, -1.0/sqrt(2), B2_A1g_color_weights_1, B2_A1g_spin_weights_1, B2_A1g_weights_1, psi2);
   make_dibaryon_correlator(C_B2_A1g, G1g_r2_BLocal, G1g_r2_BSingle, G1g_r2_BDouble, G1g_r1_BLocal, G1g_r1_BSingle, G1g_r1_BDouble, perms, sigs, -1.0/sqrt(2), B2_A1g_color_weights_2, B2_A1g_spin_weights_2, B2_A1g_weights_2, psi2);
   make_dibaryon_correlator(C_B2_T1g_r1, G1g_r1_BLocal, G1g_r1_BSingle, G1g_r1_BDouble, G1g_r1_BLocal, G1g_r1_BSingle, G1g_r1_BDouble, perms, sigs, 1.0, B2_T1g_r1_color_weights, B2_T1g_r1_spin_weights, B2_T1g_r1_weights, psi2);
   make_dibaryon_correlator(C_B2_T1g_r2, G1g_r1_BLocal, G1g_r1_BSingle, G1g_r1_BDouble, G1g_r2_BLocal, G1g_r2_BSingle, G1g_r2_BDouble, perms, sigs, 1.0/sqrt(2), B2_T1g_r2_color_weights_1, B2_T1g_r2_spin_weights_1, B2_T1g_r2_weights_1, psi2);
   make_dibaryon_correlator(C_B2_T1g_r2, G1g_r1_BLocal, G1g_r1_BSingle, G1g_r1_BDouble, G1g_r2_BLocal, G1g_r2_BSingle, G1g_r2_BDouble, perms, sigs, 1.0/sqrt(2), B2_T1g_r2_color_weights_2, B2_T1g_r2_spin_weights_2, B2_T1g_r2_weights_2, psi2);
   make_dibaryon_correlator(C_B2_T1g_r2, G1g_r2_BLocal, G1g_r2_BSingle, G1g_r2_BDouble, G1g_r1_BLocal, G1g_r1_BSingle, G1g_r1_BDouble, perms, sigs, 1.0/sqrt(2), B2_T1g_r2_color_weights_1, B2_T1g_r2_spin_weights_1, B2_T1g_r2_weights_1, psi2);
   make_dibaryon_correlator(C_B2_T1g_r2, G1g_r2_BLocal, G1g_r2_BSingle, G1g_r2_BDouble, G1g_r1_BLocal, G1g_r1_BSingle, G1g_r1_BDouble, perms, sigs, 1.0/sqrt(2), B2_T1g_r2_color_weights_2, B2_T1g_r2_spin_weights_2, B2_T1g_r2_weights_2, psi2);
   make_dibaryon_correlator(C_B2_T1g_r3, G1g_r2_BLocal, G1g_r2_BSingle, G1g_r2_BDouble, G1g_r2_BLocal, G1g_r2_BSingle, G1g_r2_BDouble, perms, sigs, 1.0, B2_T1g_r3_color_weights, B2_T1g_r3_spin_weights, B2_T1g_r3_weights, psi2);
}

int main_baryon() {
   /* indices */
   int iCprime, iSprime, jCprime, jSprime, kCprime, kSprime, iC, iS, jC, jS, kC, kS, x, x1, x2, t, y, nB1, nB2, nq, n, m;
   /* wavefunctions will be input */
   std::complex<double> psi[Nsrc][Vsrc];
   for (n=0; n<Nsrc; n++) {
      for (y=0; y<Vsrc; y++) {
         psi[n][y] = 1;
      }
   }
   std::complex<double> psi1[Nsnk][Vsnk];
   for (m=0; m<Nsnk; m++) {
      for (x=0; x<Vsnk; x++) {
         psi1[m][x] = 1;
      }
   }
   std::complex<double> psi2[Nsnk][Vsnk][Vsnk];
   for (m=0; m<Nsnk; m++) {
      for (x1=0; x1<Vsnk; x1++) {
         for (x2=0; x2<Vsnk; x2++) {
            psi2[m][x1][x2] = 1;
         }
      }
   }
   /* propagators will be input */
   std::complex<double> prop[Nq][Nc][Ns][Nc][Ns][Vsnk][Lt][Vsrc];
   for (iCprime=0; iCprime<Nc; iCprime++) {
      for (iSprime=0; iSprime<Ns; iSprime++) {
         for (iC=0; iC<Nc; iC++) {
            for (iS=0; iS<Ns; iS++) {
               for (x=0; x<Vsnk; x++) {
                  for (t=0; t<Lt; t++) {
                     for (y=0; y<Vsrc; y++) {
                        if ((iCprime == iC) && (iSprime == iS)) {
                           prop[0][iCprime][iSprime][iC][iS][x][t][y] = 1/mq;
                           prop[1][iCprime][iSprime][iC][iS][x][t][y] = 1/mq;
                           prop[2][iCprime][iSprime][iC][iS][x][t][y] = 1/mq;
                        }
                        else {
                           prop[0][iCprime][iSprime][iC][iS][x][t][y] = 0;
                           prop[1][iCprime][iSprime][iC][iS][x][t][y] = 0;
                           prop[2][iCprime][iSprime][iC][iS][x][t][y] = 0;
                        }
                     }
                  }
               }
            }
         }
      }
   }
   /* baryon (spin-color) weights will be an input */
   int B1_G1g_r1_color_weights[Nw][Nq] = { {0,1,2}, {0,2,1}, {1,0,2} ,{0,1,2}, {0,2,1}, {1,0,2}, {1,2,0}, {2,1,0}, {2,0,1} };
   int B1_G1g_r1_spin_weights[Nw][Nq] = { {0,1,0}, {0,1,0}, {0,1,0}, {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0} };
   double B1_G1g_r1_weights[Nw] = {-2/sqrt(2), 2/sqrt(2), 2/sqrt(2), 1/sqrt(2), -1/sqrt(2), -1/sqrt(2), 1/sqrt(2), -1/sqrt(2), 1/sqrt(2)};
   int B1_G1g_r2_color_weights[Nw][Nq] = { {0,1,2}, {0,2,1}, {1,0,2} ,{1,2,0}, {2,1,0}, {2,0,1}, {0,1,2}, {0,2,1}, {1,0,2} };
   int B1_G1g_r2_spin_weights[Nw][Nq] = { {0,1,1}, {0,1,1}, {0,1,1}, {0,1,1}, {0,1,1}, {0,1,1}, {1,0,1}, {1,0,1}, {1,0,1} };
   double B1_G1g_r2_weights[Nw] = {1/sqrt(2), -1/sqrt(2), -1/sqrt(2), 1/sqrt(2), -1/sqrt(2), 1/sqrt(2), -2/sqrt(2), 2/sqrt(2), 2/sqrt(2)};
   /* flavor permutations will be an input */
   int perms[Nperms][2*Nq] = { {1,2,3,4,5,6}, {1, 4, 3, 2, 5, 6}, {1, 6, 3, 2, 5, 4}, {1, 2, 3, 6, 5, 4}, {1, 4, 3, 6, 5, 2}, {1, 6, 3, 4, 5, 2}, {3, 2, 1, 4, 5, 6}, {3, 4, 1, 2, 5, 6}, {3, 6, 1, 2, 5, 4}, {3, 2, 1, 6, 5, 4}, {3, 4, 1, 6, 5, 2}, {3, 6, 1, 4, 5, 2}, {5, 2, 1, 4, 3, 6}, {5, 4, 1, 2, 3, 6}, {5, 6, 1, 2, 3, 4}, {5, 2, 1, 6, 3, 4}, {5, 4, 1, 6, 3, 2}, {5, 6, 1, 4, 3, 2}, {1, 2, 5, 4, 3, 6}, {1, 4, 5, 2, 3, 6}, {1, 6, 5, 2, 3, 4}, {1, 2, 5, 6, 3, 4}, {1, 4, 5, 6, 3, 2}, {1, 6, 5, 4, 3, 2}, {3, 2, 5, 4, 1, 6}, {3, 4, 5, 2, 1, 6}, {3, 6, 5, 2, 1, 4}, {3, 2, 5, 6, 1, 4}, {3, 4, 5, 6, 1, 2}, {3, 6, 5, 4, 1, 2}, {5, 2, 3, 4, 1, 6}, {5, 4, 3, 2, 1, 6}, {5, 6, 3, 2, 1, 4}, {5, 2, 3, 6, 1, 4}, {5, 4, 3, 6, 1, 2}, {5, 6, 3, 4, 1, 2} };
   int sigs[Nperms] = {1,-1,1,-1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,1,-1,1,-1,1};
   /* start */
   time_t start,end;
   double dif;
   time (&start);
   printf("starting block code\n");
   std::complex<double> C_pi[Lt];
   for (t=0; t<Lt; t++) {
      C_pi[t] = 0.0;
   }
   std::complex<double> C_B1_G1g_r1[Nsrc][Nsnk][Lt];
   std::complex<double> C_B1_G1g_r2[Nsrc][Nsnk][Lt];
   for (n=0; n<Nsrc; n++) {
      for (m=0; m<Nsnk; m++) {
         for (t=0; t<Lt; t++) {
            C_B1_G1g_r1[n][m][t] = 0.0;
            C_B1_G1g_r2[n][m][t] = 0.0;
         }
      }
   }
   std::complex<double> C_B2_A1g[Nsrc][Nsnk][Lt];
   std::complex<double> C_B2_T1g_r1[Nsrc][Nsnk][Lt];
   std::complex<double> C_B2_T1g_r2[Nsrc][Nsnk][Lt];
   std::complex<double> C_B2_T1g_r3[Nsrc][Nsnk][Lt];
   for (n=0; n<Nsrc; n++) {
      for (m=0; m<Nsnk; m++) {
         for (t=0; t<Lt; t++) {
            C_B2_A1g[n][m][t] = 0.0;
            C_B2_T1g_r1[n][m][t] = 0.0;
            C_B2_T1g_r2[n][m][t] = 0.0;
            C_B2_T1g_r3[n][m][t] = 0.0;
         }
      }
   }
   make_dibaryon_from_props(C_pi, C_B1_G1g_r1, C_B1_G1g_r2, C_B2_A1g, C_B2_T1g_r1, C_B2_T1g_r2, C_B2_T1g_r3, prop, B1_G1g_r1_color_weights, B1_G1g_r1_spin_weights, B1_G1g_r1_weights, B1_G1g_r2_color_weights, B1_G1g_r2_spin_weights, B1_G1g_r2_weights, perms, sigs, psi, psi1, psi2);
/*   printf("B=0 results\n");
   for (t=0; t<Lt; t++) {
      printf("C_pi[%d] = %4.9f + (%4.9f)I\n", t, creal(C_pi[t]), cimag(C_pi[t]));
   }
   printf("B=1 results\n");
   for (n=0; n<Nsrc; n++) {
      for (m=0; m<Nsnk; m++) {
         for (t=0; t<Lt; t++) {
            printf("C_B1_G1g_r1[%d][%d][%d] = %4.9f + (%4.9f)I\n", n, m, t, creal(C_B1_G1g_r1[n][m][t]), cimag(C_B1_G1g_r1[n][m][t]));
         }
      }
   }
   for (n=0; n<Nsrc; n++) {
      for (m=0; m<Nsnk; m++) {
         for (t=0; t<Lt; t++) {
            printf("C_B1_G1g_r2[%d][%d][%d] = %4.9f + (%4.9f)I\n", n, m, t, creal(C_B1_G1g_r2[n][m][t]), cimag(C_B1_G1g_r2[n][m][t]));
         }
      }
   }
   printf("B=2 results\n");
   for (n=0; n<Nsrc; n++) {
      for (m=0; m<Nsnk; m++) {
         for (t=0; t<Lt; t++) {
            printf("C_B2_A1g[%d][%d][%d] = %4.9f + (%4.9f)I\n", n, m, t, creal(C_B2_A1g[n][m][t]), cimag(C_B2_A1g[n][m][t]));
         }
      }
   }
   for (n=0; n<Nsrc; n++) {
      for (m=0; m<Nsnk; m++) {
         for (t=0; t<Lt; t++) {
            printf("C_B2_T1g_r1[%d][%d][%d] = %4.9f + (%4.9f)I\n", n, m, t, creal(C_B2_T1g_r1[n][m][t]), cimag(C_B2_T1g_r1[n][m][t]));
         }
      }
   }
   for (n=0; n<Nsrc; n++) {
      for (m=0; m<Nsnk; m++) {
         for (t=0; t<Lt; t++) {
            printf("C_B2_T1g_r2[%d][%d][%d] = %4.9f + (%4.9f)I\n", n, m, t, creal(C_B2_T1g_r2[n][m][t]), cimag(C_B2_T1g_r2[n][m][t]));
         }
      }
   }
   for (n=0; n<Nsrc; n++) {
      for (m=0; m<Nsnk; m++) {
         for (t=0; t<Lt; t++) {
            printf("C_B2_T1g_r3[%d][%d][%d] = %4.9f + (%4.9f)I\n", n, m, t, creal(C_B2_T1g_r3[n][m][t]), cimag(C_B2_T1g_r3[n][m][t]));
         }
      }
   }
   for (n=0; n<Nsrc; n++) {
      for (m=0; m<Nsnk; m++) {
         for (t=0; t<Lt; t++) {
            printf("(C_B2_A1g/(C_B1_G1g_r1*C_B1_G1g_r2)[%d][%d][%d] = %4.9f + (%4.9f)I\n", n, m, t, creal(C_B2_A1g[n][m][t]/(C_B1_G1g_r1[n][m][t]*C_B1_G1g_r2[n][m][t])), cimag(C_B2_A1g[n][m][t]/(C_B1_G1g_r1[n][m][t]*C_B1_G1g_r2[n][m][t])));
         }
      }
   }
   for (n=0; n<Nsrc; n++) {
      for (m=0; m<Nsnk; m++) {
         for (t=0; t<Lt; t++) {
           printf("(C_B2_T1g_r1/C_B1_G1g_r1^2)[%d][%d][%d] = %4.9f + (%4.9f)I\n", n, m, t, creal(C_B2_T1g_r1[n][m][t]/pow(C_B1_G1g_r1[n][m][t],2)), cimag(C_B2_T1g_r1[n][m][t]/pow(C_B1_G1g_r1[n][m][t],2)));
         }
      }
   }
   for (n=0; n<Nsrc; n++) {
      for (m=0; m<Nsnk; m++) {
         for (t=0; t<Lt; t++) {
            printf("(C_B2_T1g_r2/(C_B1_G1g_r1*C_B1_G1g_r2)[%d][%d][%d] = %4.9f + (%4.9f)I\n", n, m, t, creal(C_B2_T1g_r2[n][m][t]/(C_B1_G1g_r1[n][m][t]*C_B1_G1g_r2[n][m][t])), cimag(C_B2_T1g_r2[n][m][t]/(C_B1_G1g_r1[n][m][t]*C_B1_G1g_r2[n][m][t])));
         }
      }
   }
   for (n=0; n<Nsrc; n++) {
      for (m=0; m<Nsnk; m++) {
         for (t=0; t<Lt; t++) {
            printf("(C_B2_T1g_r3/C_B1_G1g_r2^2)[%d][%d][%d] = %4.9f + (%4.9f)I\n", n, m, t, creal(C_B2_T1g_r3[n][m][t]/pow(C_B1_G1g_r2[n][m][t],2)), cimag(C_B2_T1g_r3[n][m][t]/pow(C_B1_G1g_r2[n][m][t],2)));
         }
      }
   }
*/
   dif = difftime (end,start);
   printf("total time %5.3f\n",dif);

   return 0;
}
