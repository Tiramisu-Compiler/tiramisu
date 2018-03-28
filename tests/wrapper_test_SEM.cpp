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

  Halide::Buffer<float> D({N,N}, "D");
  Halide::Buffer<float> G({N,N,N,3,3}, "G");
  Halide::Buffer<float> u({N,N,N}, "U");
  Halide::Buffer<float> output_w({N,N,N,3}, "w");
  Halide::Buffer<float> output_dim_0(N, N, "o0");
  Halide::Buffer<float> output_dim_1(N, N, "o1");
  Halide::Buffer<float> output_dim_2(N, N, "o2");
  float f = 0.0f;
  for (int k = 0; k < N; k++) {
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
	((float*)(u.raw_buffer()->host))[i*N+j+k*N*N] = f;
	f += 1.0f;
      }
    }
  }
  f = 0.0f;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      ((float*)(D.raw_buffer()->host))[i*N+j] = f;
      f += 1.0f;
    }
  }
    
  tiramisu_timer timer;
  timer.start();
  // Run cyclops version here
  timer.stop();
  timer.print("Cyclops SEM");

  timer.start();
  tiramisu_generated_code(D.raw_buffer(), G.raw_buffer(), u.raw_buffer(), output_w.raw_buffer());
			  //			  output_dim_0.raw_buffer(),
			  //			  output_dim_1.raw_buffer(),
			  //			  output_dim_2.raw_buffer());
  timer.stop();
  timer.print("Tiramisu SEM");

  //  compare_buffers("benchmark_" + std::string(TEST_NAME_STR), output, output_ref);
  std::cerr << "D INPUT" << std::endl;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      std::cerr << ((float*)(D.raw_buffer()->host))[i*N+j] << std::endl;
    }
  }
  std::cerr << "u INPUT" << std::endl;
  for (int k = 0; k < N; k++) {
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
	std::cerr << ((float*)(u.raw_buffer()->host))[i*N+j+k*N*N] << std::endl;
      }
    }
  }
  std::cerr << "w_0, k=0" << std::endl;
  int w_idx = 0;
  int k = 0;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      std::cerr << ((float*)(output_w.raw_buffer()->host))[w_idx*N*N*N+i*N+j+k*N*N] << std::endl;
    }
  }
  std::cerr << "w_0, k=1" << std::endl;
  k = 1;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      std::cerr << ((float*)(output_w.raw_buffer()->host))[w_idx*N*N*N+i*N+j+k*N*N] << std::endl;
    }
  }
  std::cerr << "w_0, k=2" << std::endl;
  k = 2;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      std::cerr << ((float*)(output_w.raw_buffer()->host))[w_idx*N*N*N+i*N+j+k*N*N] << std::endl;
    }
  }

  std::cerr << "w_1, k=0" << std::endl;
  w_idx = 1;
  k = 0;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      std::cerr << ((float*)(output_w.raw_buffer()->host))[w_idx*N*N*N+i*N+j+k*N*N] << std::endl;
    }
  }
  std::cerr << "w_1, k=1" << std::endl;
  k = 1;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      std::cerr << ((float*)(output_w.raw_buffer()->host))[w_idx*N*N*N+i*N+j+k*N*N] << std::endl;
    }
  }
  std::cerr << "w_1, k=2" << std::endl;
  k = 2;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      std::cerr << ((float*)(output_w.raw_buffer()->host))[w_idx*N*N*N+i*N+j+k*N*N] << std::endl;
    }
  }

  std::cerr << "w_2, k=0" << std::endl;
  w_idx = 2;
  k = 0;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      std::cerr << ((float*)(output_w.raw_buffer()->host))[w_idx*N*N*N+i*N+j+k*N*N] << std::endl;
    }
  }
  std::cerr << "w_2, k=1" << std::endl;
  k = 1;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      std::cerr << ((float*)(output_w.raw_buffer()->host))[w_idx*N*N*N+i*N+j+k*N*N] << std::endl;
    }
  }
  std::cerr << "w_2, k=2" << std::endl;
  k = 2;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      std::cerr << ((float*)(output_w.raw_buffer()->host))[w_idx*N*N*N+i*N+j+k*N*N] << std::endl;
    }
  }

  return 0;
}
