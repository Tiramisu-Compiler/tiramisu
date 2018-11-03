#include "Vgg_Pencil.h"

#include <stdint.h>


void pencil_convolutionOne(float inputs[BATCH_SIZE][FIn][N1][N1],
                           float filters [FOut][FIn][pK][pK],
                           float rconv[BATCH_SIZE][FOut][N][N],
                            float bias[FOut] ) {

#pragma scop
 for (int n = 0; n < BATCH_SIZE; n++) {
        for (int f = 0; f < FOut; f++) {
            for (int y = 0; y < N; y++) {
              for (int x = 0; x < N; x++) {
                  rconv[n][f][y][x]=bias[f];}
                          }
                        }
                      }
    for (int n = 0; n < BATCH_SIZE; n++) {
        for (int f = 0; f < FOut; f++) {
            for (int y = 0; y < N; y++) {
              for (int x = 0; x < N; x++) {
                for (int fz = 0; fz < FIn; fz++) {
                   for (int fy = 0; fy < pK; fy++) {
                     for (int fx = 0; fx < pK; fx++) {
                       rconv[n][f][y][x] += filters[f][fz][fy][fx] * inputs[n][fz][y+fy][x+fx];
                                                     }
                                                   }
                                                 }
            
                                            }
                                          }
                                        }
                                        }
#pragma endscop
}
void pencil_convolutionTwo(float inputs[BATCH_SIZE][FIn][N][N],
                           float filters [FOut][FIn][pK][pK],
                           float rconv[BATCH_SIZE][FOut][N2][N2],
                           float bias[FOut] ) {

#pragma scop
 for (int n = 0; n < BATCH_SIZE; n++) {
        for (int f = 0; f < FOut; f++) {
            for (int y = 0; y < N2; y++) {
              for (int x = 0; x < N2; x++) {
                  rconv[n][f][y][x]=bias[f];}
                          }
                        }
                      }
    for (int n = 0; n < BATCH_SIZE; n++) {
        for (int f = 0; f < FOut; f++) {
            for (int y = 0; y < N2; y++) {
              for (int x = 0; x < N2; x++) {
                for (int fz = 0; fz < FIn; fz++) {
                   for (int fy = 0; fy < pK; fy++) {
                     for (int fx = 0; fx < pK; fx++) {
                       rconv[n][f][y][x] += filters[f][fz][fy][fx] * inputs[n][fz][y+fy][x+fx];
                       }
                    }
                  }
            
            }
        }
    }
}
#pragma endscop
}

void pencil_ReluOne(float inputs[BATCH_SIZE][FOut][N][N],
                float rRelu[BATCH_SIZE][FOut][N][N]){
#pragma scop
  for (int n = 0; n <BATCH_SIZE; n++) {
        for (int f = 0; f < FOut; f++) {
            for (int y = 0; y < N; y++) {
              for (int x = 0; x < N; x++) {

         rRelu[n][f][y][x]= Max(inputs[n][f][y][x],0);
                                          }
                                        }
                                      }
                                  }

#pragma endscop
}
void pencil_ReluTwo(float inputs[BATCH_SIZE][FOut][N2][N2],
                float rRelu[BATCH_SIZE][FOut][N2][N2]){
#pragma scop
  for (int n = 0; n <BATCH_SIZE; n++) {
        for (int f = 0; f < FOut; f++) {
            for (int y = 0; y < N2; y++) {
              for (int x = 0; x < N2; x++) {

         rRelu[n][f][y][x]= Max(inputs[n][f][y][x],0);
                                          }
                                        }
                                      }
                                  }

#pragma endscop
}

void pencil_MaxPooling(float inputs[BATCH_SIZE][FOut][N2][N2],
                float rPool[BATCH_SIZE][FOut][N3][N3]){

#pragma scop
for (int n = 0; n < BATCH_SIZE; n++) {
        for (int f = 0; f < FOut; f++) {
            for (int y = 0; y < N3; y++) {
              for (int x = 0; x < N3; x++) {
                rPool[n][f][y][x]=0;             
                   for (int fy = 0; fy < pK; fy++) {
                     for (int fx = 0; fx < pK; fx++) {
                          rPool[n][f][y][x]= Max(inputs[n][f][fy+y][fx+x],rPool[n][f][y][x]);
                                                    }
                                                  
                                                  }
                                            }
                                          }
                                        }
                                      }


#pragma endscop

}


void main(){

float V_filters[FOut][FIn][pK][pK];

float V_inputs[BATCH_SIZE][FIn][N1][N1];
float V_rconv[BATCH_SIZE][FOut][N][N];
float V_rRelu[BATCH_SIZE][FOut][N][N];
float V_rrconv[BATCH_SIZE][FOut][N2][N2];
float V_rrRelu[BATCH_SIZE][FOut][N2][N2];
float V_rPool[BATCH_SIZE][FOut][N3][N3];
float V_bias[FOut];
float V_bbias[FOut];

double XT[NB_TESTS];
double t1, t2, temps;
FILE* fichier = NULL;

 for(int i=0; i< NB_TESTS; i++){

//****initialisation input***// 
 for (int nn = 0; nn < BATCH_SIZE; nn++)
   for (int cc = 0; cc < FIn; cc++) 
     for (int q = 0; q < N1; q++) 
        for (int w = 0; w <N1; w++) 
          V_inputs[nn][cc][q][w]= 1;

for (int nn = 0; nn < FOut; nn++) V_bias[nn]=0;
for (int nn = 0; nn < FOut; nn++) V_bbias[nn]=0;

 for (int nn = 0; nn < BATCH_SIZE; nn++)
  for (int cc = 0; cc < FOut; cc++) 
    for (int q = 0; q < N; q++) 
        for (int w = 0; w < N; w++) 
           V_rconv[nn][cc][q][w]= 0;

 for (int nn = 0; nn < BATCH_SIZE; nn++)
  for (int cc = 0; cc < FOut; cc++) 
    for (int q = 0; q < N2; q++) 
        for (int w = 0; w < N2; w++) 
           V_rrconv[nn][cc][q][w]= 0;

 for (int nn = 0; nn < BATCH_SIZE; nn++)
  for (int cc = 0; cc < FOut; cc++) 
    for (int q = 0; q < N; q++) 
        for (int w = 0; w < N; w++) 
           V_rRelu[nn][cc][q][w]= 0;

 for (int nn = 0; nn < BATCH_SIZE; nn++)
  for (int cc = 0; cc < FOut; cc++) 
    for (int q = 0; q < N2; q++) 
        for (int w = 0; w < N2; w++) 
           V_rrRelu[nn][cc][q][w]= 0;

 for (int nn = 0; nn < BATCH_SIZE; nn++)
  for (int cc = 0; cc < FOut; cc++) 
    for (int q = 0; q < N3; q++) 
        for (int w = 0; w < N3; w++) 
           V_rPool[nn][cc][q][w]= 0;

 for (int nn = 0; nn < FOut; nn++) 
     for (int q = 0; q < FIn; q++) 
      for (int w = 0; w < pK; w++) 
        for (int s = 0; s < pK; s++) 
      V_filters[nn][q][w][s]= 1;

//**** VGG Bloc***// 
  t1 = rtclock();

  //First Conv
  pencil_convolutionOne(V_inputs,
             V_filters,
             V_rconv, V_bias);

  //First Relu
   pencil_ReluOne(V_rconv,
             V_rRelu);
  //Second Conv
     pencil_convolutionTwo(V_rRelu,
             V_filters,
             V_rrconv, V_bbias);

  //Second Relu
      pencil_ReluTwo(V_rrconv,
             V_rrRelu);

  // Max pooling
      pencil_MaxPooling(V_rrRelu,
               V_rPool);

  // Time

    t2 = rtclock();
   temps = (double)(t2-t1);
   if (NB_TESTS>1)  XT[i]= temps;
 }

 if (NB_TESTS>1)  temps= median(NB_TESTS, XT);
     printf("%f seconds\n",temps);

fichier = fopen("vgg_pencil_result.txt", "w");
if (fichier != NULL)

    { 
       
       for (int nn = 0; nn < BATCH_SIZE; nn++)
       for (int cc = 0; cc < FOut; cc++) 
       for (int q = 0; q < N3; q++) 
       for (int w = 0; w < N3; w++) 
       fprintf(fichier, "%.0f", V_rPool[nn][cc][q][w]);
       fclose(fichier);
     
    }

    else printf("Impossible d'ouvrir le fichier test.txt");


}
