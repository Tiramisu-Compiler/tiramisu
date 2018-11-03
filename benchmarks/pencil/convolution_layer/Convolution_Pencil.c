#include "convolution_Pencil.h"

#include <stdint.h>

void pencil_convolution(float inputs[BATCH_SIZE][FIn][N1][N1],
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

void main(){

float C_inputs[BATCH_SIZE][FIn][N1][N1];
float C_filters [FOut][FIn][pK][pK];
float C_rconv[BATCH_SIZE][FOut][N][N];
float C_bias[FOut];
double XT[NB_TESTS];
double t1, t2, temps;
FILE* fichier = NULL;

 for(int i=0; i< NB_TESTS; i++){

//****initialisation input***// 
 for (int nn = 0; nn < BATCH_SIZE; nn++)
   for (int cc = 0; cc < FIn; cc++) 
     for (int q = 0; q < N1; q++) 
        for (int w = 0; w <N1; w++) 
          C_inputs[nn][cc][q][w]= 1;

 for (int nn = 0; nn < FOut; nn++) C_bias[nn]=1;

 for (int nn = 0; nn < BATCH_SIZE; nn++)
  for (int cc = 0; cc < FOut; cc++) 
    for (int q = 0; q < N; q++) 
        for (int w = 0; w < N; w++) 
           C_rconv[nn][cc][q][w]= 0;

 for (int nn = 0; nn < FOut; nn++) 
     for (int q = 0; q < FIn; q++) 
      for (int w = 0; w < pK; w++) 
        for (int s = 0; s < pK; s++) 
      C_filters[nn][q][w][s]= 1;



 t1 = rtclock();
pencil_convolution(C_inputs,
             C_filters,
             C_rconv,C_bias);

    t2 = rtclock();
   temps = (double)(t2-t1);
   if (NB_TESTS>1)  XT[i]= temps;
 }

 if (NB_TESTS>1)  temps= median(NB_TESTS, XT);

  printf("%f seconds\n",temps);
// Display results
fichier = fopen("convolution_layer_pencil_result.txt", "w");
if (fichier != NULL)

    { 
      
       for (int nn = 0; nn < BATCH_SIZE; nn++){
       for (int cc = 0; cc < FOut; cc++){ 
       for (int q = 0; q < N; q++) {
       for (int w = 0; w < N; w++){  
       fprintf(fichier, "%.0f", C_rconv[nn][cc][q][w]);
                }
              }
             }
           }
       fclose(fichier);

    }

    else printf("Impossible d'ouvrir le fichier test.txt");


}


