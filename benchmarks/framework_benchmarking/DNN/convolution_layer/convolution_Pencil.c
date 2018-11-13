#include "../config.h"


  void pencil_convolution(float inputs[BATCH_SIZE][FIn][N1][N1],
    float filters[FOut][FIn][pK][pK],
    float rconv[BATCH_SIZE][FOut][N][N],
    float bias[FOut]) {

    #pragma scop
    for (int n = 0; n < BATCH_SIZE; n++) {
      for (int f = 0; f < FOut; f++) {
        for (int y = 0; y < N; y++) {
          for (int x = 0; x < N; x++) {
            rconv[n][f][y][x] = bias[f];
          }
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
                  rconv[n][f][y][x] += filters[f][fz][fy][fx] * inputs[n][fz][y + fy][x + fx];
                }
              }
            }

          }
        }
      }
    }
#pragma endscop
  }

void main() {

  int batch_size = BATCH_SIZE;
  int fin = FIn;
  int n1 = N1;
  int pk = pK;
  int fout = FOut;
  int n = N;
  long int SizeInput = batch_size * fin * n1 * n1;
  long int SizeFilter = fout * fin * pk * pk;
  long int SizeConv = batch_size * fout * n * n;

  float * C_inputs = NULL;
  C_inputs = malloc(SizeInput * sizeof(float));

  float * C_filters = NULL;
  C_filters = malloc(SizeFilter * sizeof(float));

  float * C_rconv = NULL;
  C_rconv = malloc(SizeConv * sizeof(float));

  float * C_bias = NULL;
  C_bias = malloc(fout * sizeof(float));

  double XT[NB_TESTS];
  double t1, t2, temps;
  FILE * fichier = NULL;

  for (int i = 0; i < NB_TESTS; i++) {

    //****initialisation input***// 

    for (int nn = 0; nn < SizeInput; nn++) C_inputs[nn] = 1;
    for (int nn = 0; nn < fout; nn++) C_bias[nn] = 1;
    for (int nn = 0; nn < SizeConv; nn++) C_rconv[nn] = 0;
    for (int nn = 0; nn < SizeFilter; nn++) C_filters[nn] = 1;

    t1 = rtclock();

    pencil_convolution(C_inputs,
      C_filters,
      C_rconv, C_bias);

    t2 = rtclock();
    temps = (double)(t2 - t1);
    if (NB_TESTS > 1) XT[i] = temps;
  }

  if (NB_TESTS > 1) temps = median(NB_TESTS, XT);
  printf("Execution time:  %f ms\n", temps);

  // Display results
  fichier = fopen("convolution_layer_pencil_result.txt", "w");
  if (fichier != NULL)

  {
    for (int nn = 0; nn < SizeConv; nn++) fprintf(fichier, "%.0f", C_rconv[nn]);
    fclose(fichier);
  } else printf("Error: cannot open the result file");

  free(C_inputs);
  free(C_filters);
  free(C_rconv);
  free(C_bias);
}
