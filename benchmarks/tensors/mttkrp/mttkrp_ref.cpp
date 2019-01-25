#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>


#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}  // extern "C"
#endif

void ref(double A_vals[SIZE][SIZE],
	 double B_vals[SIZE][SIZE][SIZE],
	 double C_vals[SIZE][SIZE],
	 double D_vals[SIZE][SIZE])
{
    for (int32_t iB = 0; iB < SIZE; iB++)
      for (int32_t jD = 0; jD < SIZE; jD++)
        A_vals[iB][jD] = 0.0;

    for (int32_t iB = 0; iB < SIZE; iB++)
      for (int32_t kB = 0; kB < SIZE; kB++)
	for (int32_t lB = 0; lB < SIZE; lB++) {
	  double tl = B_vals[iB][kB][lB];
	  for (int32_t jD = 0; jD < SIZE; jD++)
	    A_vals[iB][jD] = A_vals[iB][jD] + tl * D_vals[lB][jD] * C_vals[kB][jD];
	}
}

void init_buffers(double A_vals[SIZE][SIZE],
	 	  double B_vals[SIZE][SIZE][SIZE],
	 	  double C_vals[SIZE][SIZE],
	 	  double D_vals[SIZE][SIZE])
{
    for (int32_t iB = 0; iB < SIZE; iB++)
      for (int32_t jD = 0; jD < SIZE; jD++)
        A_vals[iB][jD] = iB + jD;

    for (int32_t iB = 0; iB < SIZE; iB++)
      for (int32_t kB = 0; kB < SIZE; kB++)
	for (int32_t lB = 0; lB < SIZE; lB++)
	  B_vals[iB][kB][lB] = iB + kB + lB;

    for (int32_t kB = 0; kB < SIZE; kB++)
      for (int32_t jD = 0; jD < SIZE; jD++)
	 C_vals[kB][jD] = kB + jD;

    for (int32_t lB = 0; lB < SIZE; lB++)
      for (int32_t jD = 0; jD < SIZE; jD++)
	  D_vals[lB][jD] = lB + jD;
}
