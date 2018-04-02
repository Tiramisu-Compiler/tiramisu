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
  Halide::Buffer<double> u({N,N,N}, "U");

  tiramisu_timer timer;
  timer.start();
  tiramisu_generated_code(D.raw_buffer(), G.raw_buffer(), u.raw_buffer());
  timer.stop();
  timer.print("Tiramisu SEM");
  
  return 0;
}
