#include "Halide.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>

#include "wrapper_test_SEM.h"


#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}  // extern "C"
#endif

int main(int, char **) {

  Halide::Buffer<double> D({N,N}, "D");
  Halide::Buffer<double> G({N,N,N,3,3}, "G");
  Halide::Buffer<double> u({N,N,N}, "u");
#if defined(PRINT_RES) || defined(UNOPT)
  double f = 0.0;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
	((double*)(u.raw_buffer()->host))[i*N*N + j*N + k] = f;
	f += 0.01;
      }
    }
  }
  f = 0.0;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
	((double*)(D.raw_buffer()->host))[i*N+ j] = f;
	f += 0.01;
    }
  }
  f = 0.0;
  for (int a = 0; a < 3; a++) {
    for (int b = 0; b < 3; b++) {
      for (int i = 0; i < N; i++) {
	for (int j = 0; j < N; j++) {
	  for (int k = 0; k < N; k++) {
	    ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + i*N*N + j*N + k] = f;
	    f += 0.01;
	  }
	}
      }
    }
  }
#endif
  
  tiramisu_timer timer;
  timer.start();
  tiramisu_generated_code(D.raw_buffer(), G.raw_buffer(), u.raw_buffer());
  timer.stop(); 
  timer.print("Tiramisu SEM");
  
#if defined(PRINT_RES) || defined(UNOPT)
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
	std::cerr << ((double*)(u.raw_buffer()->host))[i*N*N + j*N + k] << std::endl;
      }
    }
  }
#endif
  return 0;
}
