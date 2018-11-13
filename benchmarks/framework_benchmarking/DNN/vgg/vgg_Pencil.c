#include "../config.h"

  void pencil_vgg(float inputs[BATCH_SIZE][FIn][N1][N1],
    float filters[FOut][FIn][pK][pK],
    float rconv[BATCH_SIZE][FOut][N][N],
    float bias[FOut],
    float rRelu[BATCH_SIZE][FOut][N][N],
    float rrconv[BATCH_SIZE][FOut][N2][N2],
    float rrRelu[BATCH_SIZE][FOut][N2][N2],
    float rPool[BATCH_SIZE][FOut][N3][N3]) {

    #pragma scop
    //First Conv
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

    //First Relu
    for (int n = 0; n < BATCH_SIZE; n++) {
      for (int f = 0; f < FOut; f++) {
        for (int y = 0; y < N; y++) {
          for (int x = 0; x < N; x++) {

            rRelu[n][f][y][x] = Max(rconv[n][f][y][x], 0);
          }
        }
      }
    }

    //Second Conv
    for (int n = 0; n < BATCH_SIZE; n++) {
      for (int f = 0; f < FOut; f++) {
        for (int y = 0; y < N2; y++) {
          for (int x = 0; x < N2; x++) {
            rrconv[n][f][y][x] = bias[f];
          }
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
                  rrconv[n][f][y][x] += filters[f][fz][fy][fx] * rRelu[n][fz][y + fy][x + fx];
                }
              }
            }

          }
        }
      }
    }

    //Second Relu
    for (int n = 0; n < BATCH_SIZE; n++) {
      for (int f = 0; f < FOut; f++) {
        for (int y = 0; y < N2; y++) {
          for (int x = 0; x < N2; x++) {

            rrRelu[n][f][y][x] = Max(rrconv[n][f][y][x], 0);
          }
        }
      }
    }

    // Max pooling
    for (int n = 0; n < BATCH_SIZE; n++) {
      for (int f = 0; f < FOut; f++) {
        for (int y = 0; y < N3; y++) {
          for (int x = 0; x < N3; x++) {
            rPool[n][f][y][x] = 0;
            for (int fy = 0; fy < pK; fy++) {
              for (int fx = 0; fx < pK; fx++) {
                rPool[n][f][y][x] = Max(rrRelu[n][f][fy + y][fx + x], rPool[n][f][y][x]);
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
  int n2 = N2;
  int n3 = N3;
  long int SizeInput = batch_size * fin * n1 * n1;
  long int SizeFilter = fout * fin * pk * pk;
  long int SizeConv = batch_size * fout * n * n;
  long int SizeRrconv = batch_size * fout * n2 * n2;
  long int SizePool = batch_size * fout * n3 * n3;

  float * V_inputs = NULL;
  V_inputs = malloc(SizeInput * sizeof(float));

  float * V_filters = NULL;
  V_filters = malloc(SizeFilter * sizeof(float));

  float * V_rconv = NULL;
  V_rconv = malloc(SizeConv * sizeof(float));

  float * V_rRelu = NULL;
  V_rRelu = malloc(SizeConv * sizeof(float));

  float * V_rrconv = NULL;
  V_rrconv = malloc(SizeRrconv * sizeof(float));

  float * V_rrRelu = NULL;
  V_rrRelu = malloc(SizeRrconv * sizeof(float));

  float * V_rPool = NULL;
  V_rPool = malloc(SizePool * sizeof(float));

  float * V_bias = NULL;
  V_bias = malloc(fout * sizeof(float));

  float * V_bbias = NULL;
  V_bbias = malloc(fout * sizeof(float));

  double XT[NB_TESTS];
  double t1, t2, temps;
  FILE * fichier = NULL;

  for (int i = 0; i < NB_TESTS; i++) {

    //****initialisation input***// 
    for (int nn = 0; nn < SizeInput; nn++) V_inputs[nn] = 1;

    for (int nn = 0; nn < fout; nn++) V_bias[nn] = 0;
    for (int nn = 0; nn < fout; nn++) V_bbias[nn] = 0;

    for (int nn = 0; nn < SizeConv; nn++) V_rconv[nn] = 0;

    for (int nn = 0; nn < SizeRrconv; nn++) V_rrconv[nn] = 0;

    for (int nn = 0; nn < SizeConv; nn++) V_rRelu[nn] = 0;

    for (int nn = 0; nn < SizeRrconv; nn++) V_rrRelu[nn] = 0;

    for (int nn = 0; nn < SizePool; nn++) V_rPool[nn] = 0;

    for (int nn = 0; nn < SizeFilter; nn++) V_filters[nn] = 1;

    //**** VGG Bloc***// 
    t1 = rtclock();

    pencil_vgg(V_inputs, V_filters, V_rconv, V_bias,
      V_rRelu,
      V_rrconv,
      V_rrRelu,
      V_rPool);
    t2 = rtclock();
    //**** Time***// 
    temps = (double)(t2 - t1);
    if (NB_TESTS > 1) XT[i] = temps;
  }

  if (NB_TESTS > 1) temps = median(NB_TESTS, XT);
  printf("Execution time: %f ms\n", temps);

  fichier = fopen("vgg_pencil_result.txt", "w");
  if (fichier != NULL)

  {

    for (int nn = 0; nn < SizePool; nn++) fprintf(fichier, "%.0f", V_rPool[nn]);
    fclose(fichier);

  } else printf("Error: cannot open the result file");

}
