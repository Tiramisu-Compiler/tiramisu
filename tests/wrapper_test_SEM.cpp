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
  Halide::Buffer<double> tmp_buff({N,N,N,3}, "tmp");

  ((double*)(u.raw_buffer()->host))[0] = 0.159793;
  ((double*)(u.raw_buffer()->host))[1] = 0.542285;
  ((double*)(u.raw_buffer()->host))[2] = 0.039569;
  ((double*)(u.raw_buffer()->host))[3] = 0.631528;
  ((double*)(u.raw_buffer()->host))[4] = 0.992145;
  ((double*)(u.raw_buffer()->host))[5] = 0.0571598;
  ((double*)(u.raw_buffer()->host))[6] = 0.597495;
  ((double*)(u.raw_buffer()->host))[7] = 0.423571;

  /*std::cerr << "***U***" << std::endl;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
	std::cerr << ((double*)(u.raw_buffer()->host))[i*N*N+j*N+k] << std::endl;
      }
    }
    }*/
  
  ((double*)(D.raw_buffer()->host))[0] = 0.842261;
  ((double*)(D.raw_buffer()->host))[1] = 0.423148;
  ((double*)(D.raw_buffer()->host))[2] = 0.906301;
  ((double*)(D.raw_buffer()->host))[3] = 0.655526;
  
  int a = 0, b = 0;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 0] = 0.930431;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 1] = 0.790189;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 2] = 0.254615;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 3] = 0.557681;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 4] = 0.344386;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 5] = 0.69542;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 6] = 0.444671;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 7] = 0.28743;
  b = 1;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 0] = 0.145816;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 1] = 0.150165;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 2] = 0.56455;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 3] = 0.554408;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 4] = 0.579264;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 5] = 0.0199339;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 6] = 0.146418;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 7] = 0.687332;
  b = 2;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 0] = 0.388153;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 1] = 0.476169;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 2] = 0.179794;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 3] = 0.980407;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 4] = 0.928789;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 5] = 0.25709;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 6] = 0.189908;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 7] = 0.985471;
  a = 1; b = 0;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 0] = 0.159543;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 1] = 0.172558;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 2] = 0.767771;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 3] = 0.31673;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 4] = 0.670065;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 5] = 0.13003;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 6] = 0.838183;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 7] = 0.882861;
  b = 1;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 0] = 0.399804;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 1] = 0.115438;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 2] = 0.51594;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 3] = 0.143895;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 4] = 0.417875;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 5] = 0.757875;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 6] = 0.495044;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 7] = 0.214071;
  b = 2;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 0] = 0.195208;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 1] = 0.529509;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 2] = 0.842965;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 3] = 0.533065;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 4] = 0.595918;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 5] = 0.540282;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 6] = 0.863895;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 7] = 0.116011;
  a = 2; b = 0;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 0] = 0.950275;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 1] = 0.301448;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 2] = 0.278345;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 3] = 0.0698625;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 4] = 0.797556;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 5] = 0.00273905;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 6] = 0.553107;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 7] = 0.198198;
  b = 1;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 0] = 0.0472926;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 1] = 0.773028;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 2] = 0.892334;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 3] = 0.0751163;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 4] = 0.413537;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 5] = 0.693171;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 6] = 0.408317;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 7] = 0.00563984;
  b = 2;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 0] = 0.274555;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 1] = 0.226628;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 2] = 0.903401;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 3] = 0.600954;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 4] = 0.673547;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 5] = 0.803385;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 6] = 0.330416;
  ((double*)(G.raw_buffer()->host))[a*3*N*N*N + b*N*N*N + 7] = 0.748014;
  
  /*  std::cerr << "*** G INPUT ***" << std::endl;
  for (int a = 0; a < 3; a++) {
    for (int b = 0; b < 3; b++) {
      for (int i = 0; i < N; i++) {
	for (int j = 0; j < N; j++) {
	  for (int k = 0; k < N; k++) {
	    std::cerr << ((double*)(G.raw_buffer()->host))[a*3*N*N*N+
							   b*N*N*N+
							   i*N*N+
							   j*N+k] << std::endl;
	  }
	}
      }
    }
  }*/

  tiramisu_timer timer;
  timer.start();
  tiramisu_generated_code(D.raw_buffer(), G.raw_buffer(), u.raw_buffer(), tmp_buff.raw_buffer());
  timer.stop();
  timer.print("Tiramisu SEM");


  std::cerr << "***TMP OUTPUT***" << std::endl;
  for (int t = 0; t < 3; t++) {
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
	for (int k = 0; k < N; k++) {
	std::cerr << ((double*)(tmp_buff.raw_buffer()->host))[t*N*N*N + i*N*N + j*N + k] << std::endl;
	}
      }
    }
    std::cerr << "***" << std::endl;
  }

  std::cerr << "***D***" << std::endl;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      std::cerr << ((double*)(D.raw_buffer()->host))[i*N+j] << std::endl;
    }
  }
  
  /*  std::cerr << "***U OUTPUT***" << std::endl;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
	std::cerr << ((double*)(u.raw_buffer()->host))[i*N*N+j*N+k] << std::endl;
      }
    }
    }*/

  
  return 0;
}
