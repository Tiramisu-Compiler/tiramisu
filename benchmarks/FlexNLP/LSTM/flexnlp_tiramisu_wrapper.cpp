#include "generated_flexnlp_test.o.h"

#include "Halide.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>
#include "configure.h"

int main(int, char **)
{
  std::vector<double> duration_vector;
  double start, end;

  const int hsize = HIDDEN_SIZE;
  const int isize = INPUT_SIZE;
  const int osize = hsize;
  const int nbatch = BATCH_SIZE;
  const int nlayers = NUM_LAYERS;
  const int ntimestep = SEQ_LENGTH;
  // test LSTM Layer,
  int8_t* x_in = (int8_t*) malloc(ntimestep*nbatch*isize*sizeof(int8_t));
  int8_t* w_in = (int8_t*) malloc(nlayers*4*osize*(isize+hsize)*sizeof(int8_t));
  int8_t* output = (int8_t*) malloc(ntimestep*nbatch*hsize*sizeof(int8_t));
  int8_t* h_out = (int8_t*) malloc(nlayers*nbatch*hsize*sizeof(int8_t));

  for(int i=0; i<ntimestep*nbatch*isize; i++)
    if (i%3 == 0)
      x_in[i] = (int8_t) 2;
    else if (i%3 == 1)
      x_in[i] = (int8_t) -3;
    else
      x_in[i] = (int8_t) 100;


  for(int i=0; i<nlayers*4*osize*(isize+hsize); i++)
    if (i%3 == 0)
      w_in[i] = (int8_t) 2;
    else if (i%3 == 1)
      w_in[i] = (int8_t) -3;
    else
      w_in[i] = (int8_t) 100;

  for(int i=0; i<nlayers*nbatch*hsize; i++)
    h_out[i] = (int8_t) 3;

  for(int i=0; i<ntimestep*nbatch*hsize; i++)
    output[i] = (int8_t) 0;

  Halide::Buffer<int8_t> b_input(x_in, ntimestep, nbatch, isize);
  Halide::Buffer<int8_t> b_W(w_in, NUM_LAYERS, 4, osize, isize + hsize);
  Halide::Buffer<int8_t> b_output(output, ntimestep, nbatch, hsize);
  Halide::Buffer<int8_t> b_h_out(h_out, nlayers, nbatch, hsize);

  std::cout << "Buffers Initialized" << std::endl;

  for (int i = 0; i < NB_TESTS; i++)
  {
    for(int i=0; i<nlayers*nbatch*hsize; i++)
      h_out[i] = (int8_t) 3;

    for(int i=0; i<ntimestep*nbatch*hsize; i++)
      output[i] = (int8_t) 0;

    start = rtclock();
    flexnlp_lstm(
      b_input.raw_buffer(),
      b_W.raw_buffer(),
      b_output.raw_buffer(),
      b_h_out.raw_buffer()
    );
    end = rtclock();
    duration_vector.push_back((end - start) * 1000);
  }

  print_time("performance_CPU.csv", "FlexLSTM",
             {"Tiramisu"},
             {median(duration_vector)});

  for(int i = 0; i<100; i++){
    std::cout << " , " << (int) output[i];
  }
  std::cout << "Finished" << std::endl;

  return 0;
}
