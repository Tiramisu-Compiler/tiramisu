#include "convolution_Pencil.h"

#include <stdint.h>

void pencil_convolution(float inputs[BATCH_SIZE][FIn][N1][N1],
                           float filters [FOut][FIn][pK][pK],
                           float rconv[BATCH_SIZE][FOut][N][N],
                            float bias[FOut] ) {

 /* ppcg generated CPU code */
 
 #pragma omp parallel for
 for (int c0 = 0; c0 <= 7; c0 += 1)
   for (int c1 = 0; c1 <= 15; c1 += 1)
     for (int c2 = 0; c2 <= 31; c2 += 1)
       for (int c3 = 0; c3 <= 31; c3 += 1) {
         rconv[c0][c1][c2][c3] = bias[c1];
         for (int c4 = 0; c4 <= 15; c4 += 1)
           for (int c5 = 0; c5 <= 4; c5 += 1)
             for (int c6 = 0; c6 <= 4; c6 += 1)
               rconv[c0][c1][c2][c3] += (filters[c1][c4][c5][c6] * inputs[c0][c4][c2 + c5][c3 + c6]);
       }
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


