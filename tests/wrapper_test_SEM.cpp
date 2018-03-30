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
  Halide::Buffer<float> output_D_dim_i(N, "o0");
  Halide::Buffer<float> output_D_dim_j(N, "o1");
  Halide::Buffer<float> output_u_dim_i({N,N}, "ui");
  Halide::Buffer<float> output_u_dim_j({N,N}, "uj");
  Halide::Buffer<float> output_u_dim_k({N,N}, "uk");
  Halide::Buffer<float> output_z({N,N,N,3}, "z");
  
  float f = 0.0f;
  for (int a = 0; a < 3; a++) {
    for (int b = 0; b < 3; b++) {
      for (int i = 0; i < N; i++) {
	for (int j = 0; j < N; j++) {
	  for (int k = 0; k < N; k++) {
	    ((float*)(G.raw_buffer()->host))[a*3*N*N*N+
					     b*N*N*N+
					     i*N*N+
					     j*N+
					     k] = f;
	    f += 1.0f;
	  }
	}
      }
    }
  }
  f = 0.0f;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
	((float*)(u.raw_buffer()->host))[i*N*N+j*N+k] = f;
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
  tiramisu_generated_code(D.raw_buffer(), G.raw_buffer(), u.raw_buffer(),
			  output_D_dim_i.raw_buffer(), output_D_dim_j.raw_buffer(), output_w.raw_buffer(),
			  output_u_dim_i.raw_buffer(), output_u_dim_j.raw_buffer(), output_u_dim_k.raw_buffer(),
			  output_z.raw_buffer());
  timer.stop();
  timer.print("Tiramisu SEM");

  //  compare_buffers("benchmark_" + std::string(TEST_NAME_STR), output, output_ref);
  std::cerr << "*** G INPUT ***" << std::endl;
  for (int a = 0; a < 3; a++) {
    for (int b = 0; b < 3; b++) {
      for (int i = 0; i < N; i++) {
	for (int j = 0; j < N; j++) {
	  for (int k = 0; k < N; k++) {
	    std::cerr << ((float*)(G.raw_buffer()->host))[a*3*N*N*N+
							  b*N*N*N+
							  i*N*N+
							  j*N+k] << std::endl;
	  }
	}
      }
    }
  }
  std::cerr << "*** D input ***" << std::endl;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      std::cerr << ((float*)(D.raw_buffer()->host))[i*N+j] << std::endl;
    }
  }
  std::cerr << "*** D reduction along dimension i (0) ***" << std::endl;
  for (int j = 0; j < N; j++) {
    std::cerr << ((float*)(output_D_dim_i.raw_buffer()->host))[j] << std::endl;
  }
  std::cerr << "*** D reduction along dimension j (1) ***" << std::endl;
  for (int i = 0; i < N; i++) {
    std::cerr << ((float*)(output_D_dim_j.raw_buffer()->host))[i] << std::endl;
  }
  std::cerr << "*** u input ***" << std::endl;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
	std::cerr << ((float*)(u.raw_buffer()->host))[i*N*N+j*N+k] << std::endl;
      }
    }
  }
  std::cerr << "*** u reduction along dimension i (0) ***" << std::endl;
  for (int j = 0; j < N; j++) {
    for (int k = 0; k < N; k++) {
      std::cerr << ((float*)(output_u_dim_i.raw_buffer()->host))[j*N+k] << std::endl;
    }
  }
  std::cerr << "*** u reduction along dimension j (1) ***" << std::endl;
  for (int i = 0; i < N; i++) {
    for (int k = 0; k < N; k++) {
      std::cerr << ((float*)(output_u_dim_j.raw_buffer()->host))[i*N+k] << std::endl;
    }
  }  
  std::cerr << "*** u reduction along dimension k (2) ***" << std::endl;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      std::cerr << ((float*)(output_u_dim_k.raw_buffer()->host))[i*N+j] << std::endl;
    }
  }  
  std::cerr << "*** w_0 ***" << std::endl;
  int w_idx = 0;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
	std::cerr << ((float*)(output_w.raw_buffer()->host))[w_idx*N*N*N+i*N*N+j*N+k] << std::endl;
      }
    }
  }
  std::cerr << "*** w_1 ***" << std::endl;
  w_idx = 1;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
	std::cerr << ((float*)(output_w.raw_buffer()->host))[w_idx*N*N*N+i*N*N+j*N+k] << std::endl;
      }
    }
  }
  std::cerr << "*** w_2 ***" << std::endl;
  w_idx = 2;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
	std::cerr << ((float*)(output_w.raw_buffer()->host))[w_idx*N*N*N+i*N*N+j*N+k] << std::endl;
      }
    }
  }
  std::cerr << "*** z ***" << std::endl;
  for (int a = 0; a < 3; a++) {
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
	for (int k = 0; k < N; k++) {
	  std::cerr << ((float*)(output_z.raw_buffer()->host))[a*N*N*N+i*N*N+j*N+k] << " ";
	}
      }
    }
    std::cerr << std::endl;
  }
  std::cerr << "*** u result ***" << std::endl;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
	std::cerr << ((float*)(u.raw_buffer()->host))[i*N*N+j*N+k] << std::endl;
      }
    }
  }  
  return 0;
}
