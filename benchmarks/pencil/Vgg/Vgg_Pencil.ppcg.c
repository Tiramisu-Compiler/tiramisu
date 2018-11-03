#include "Vgg_Pencil.h"

#include <stdint.h>


void pencil_convolutionOne(float inputs[BATCH_SIZE][FIn][N1][N1],
                           float filters [FOut][FIn][pK][pK],
                           float rconv[BATCH_SIZE][FOut][N][N],
                            float bias[FOut] ) {

 /* ppcg generated CPU code */
 
 #pragma omp parallel for
 for (int c0 = 0; c0 <= 7; c0 += 1)
   for (int c1 = 0; c1 <= 3; c1 += 1)
     for (int c2 = 0; c2 <= 31; c2 += 1)
       for (int c3 = 0; c3 <= 31; c3 += 1) {
         rconv[c0][c1][c2][c3] = bias[c1];
         for (int c4 = 0; c4 <= 3; c4 += 1)
           for (int c5 = 0; c5 <= 4; c5 += 1)
             for (int c6 = 0; c6 <= 4; c6 += 1)
               rconv[c0][c1][c2][c3] += (filters[c1][c4][c5][c6] * inputs[c0][c4][c2 + c5][c3 + c6]);
       }
}
void pencil_convolutionTwo(float inputs[BATCH_SIZE][FIn][N][N],
                           float filters [FOut][FIn][pK][pK],
                           float rconv[BATCH_SIZE][FOut][N2][N2],
                           float bias[FOut] ) {

 /* ppcg generated CPU code */
 
 #pragma omp parallel for
 for (int c0 = 0; c0 <= 7; c0 += 1)
   for (int c1 = 0; c1 <= 3; c1 += 1)
     for (int c2 = 0; c2 <= 27; c2 += 1)
       for (int c3 = 0; c3 <= 27; c3 += 1) {
         rconv[c0][c1][c2][c3] = bias[c1];
         for (int c4 = 0; c4 <= 3; c4 += 1)
           for (int c5 = 0; c5 <= 4; c5 += 1)
             for (int c6 = 0; c6 <= 4; c6 += 1)
               rconv[c0][c1][c2][c3] += (filters[c1][c4][c5][c6] * inputs[c0][c4][c2 + c5][c3 + c6]);
       }
}

void pencil_ReluOne(float inputs[BATCH_SIZE][FOut][N][N],
                float rRelu[BATCH_SIZE][FOut][N][N]){
  /* ppcg generated CPU code */
  
  #pragma omp parallel for
  for (int c0 = 0; c0 <= 7; c0 += 1)
    for (int c1 = 0; c1 <= 3; c1 += 1)
      for (int c2 = 0; c2 <= 31; c2 += 1)
        for (int c3 = 0; c3 <= 31; c3 += 1)
          rRelu[c0][c1][c2][c3] = Max(inputs[c0][c1][c2][c3], 0);
}
void pencil_ReluTwo(float inputs[BATCH_SIZE][FOut][N2][N2],
                float rRelu[BATCH_SIZE][FOut][N2][N2]){
  /* ppcg generated CPU code */
  
  #pragma omp parallel for
  for (int c0 = 0; c0 <= 7; c0 += 1)
    for (int c1 = 0; c1 <= 3; c1 += 1)
      for (int c2 = 0; c2 <= 27; c2 += 1)
        for (int c3 = 0; c3 <= 27; c3 += 1)
          rRelu[c0][c1][c2][c3] = Max(inputs[c0][c1][c2][c3], 0);
}

void pencil_MaxPooling(float inputs[BATCH_SIZE][FOut][N2][N2],
                float rPool[BATCH_SIZE][FOut][N3][N3]){

/* ppcg generated CPU code */

#pragma omp parallel for
for (int c0 = 0; c0 <= 7; c0 += 1)
  for (int c1 = 0; c1 <= 3; c1 += 1)
    for (int c2 = 0; c2 <= 23; c2 += 1)
      for (int c3 = 0; c3 <= 23; c3 += 1) {
        rPool[c0][c1][c2][c3] = 0;
        for (int c4 = 0; c4 <= 4; c4 += 1)
          for (int c5 = 0; c5 <= 4; c5 += 1)
            rPool[c0][c1][c2][c3] = Max(inputs[c0][c1][c2 + c4][c3 + c5], rPool[c0][c1][c2][c3]);
      }

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
