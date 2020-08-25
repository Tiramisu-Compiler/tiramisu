#include "generated_flexnlp_test.o.h"

#include "Halide.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>
#include "configure.h"
#include "../../../../src/FlexNLP/LSTM.h"
#include "../../../../src/FlexNLP/Utils.h"

int main(int, char **)
{
  // Executing the reference code
  int i_size = INPUT_SIZE;
  int h_size = FEATURE_SIZE;

  std::cout << "Running reference LSTM version"<< std::endl;
  // Software and Reference output
  LSTM lstms[NUM_LAYERS];
  lstms[0] =  LSTM(i_size, h_size, true);
  lstms[0].random_state();

  for (int i = 1; i < NUM_LAYERS; i++){
    lstms[i] =  LSTM(h_size, h_size, true);
    lstms[i].random_state();
  }
  std::vector<std::vector<float>> Wx[NUM_LAYERS];
  std::vector<std::vector<float>> Wh[NUM_LAYERS];
  std::vector<float> bx[NUM_LAYERS];
  std::vector<float> bh[NUM_LAYERS];

  std::vector<float> x_in[SEQ_LENGTH];
  std::vector<float> h_in[NUM_LAYERS];
  std::vector<float> c_in[NUM_LAYERS];

  for (int i = 0; i < NUM_LAYERS; i++){
    // Software weight/bias
    Wx[i] = lstms[i].GetWx();
    Wh[i] = lstms[i].GetWh();
    bx[i] = lstms[i].Getbx();
    bh[i] = lstms[i].Getbh();
    h_in[i] = lstms[i].h_t;
    c_in[i] = lstms[i].c_t;
  }
  for (int i = 0; i < SEQ_LENGTH; i++){
    x_in[i] = GetRandVec(i_size);
  }

  // Run reference LSTM
  std::vector<std::vector<float>> h_out_ref[NUM_LAYERS];
  std::vector<float> c_out_ref[NUM_LAYERS];
  for (int l = 0; l < NUM_LAYERS; l++){
    if (l==0)
      for (int s = 0; s < SEQ_LENGTH; s++){
        h_out_ref[l].push_back(lstms[l].run(x_in[s]));
      }
    else
      for (int s = 0; s < SEQ_LENGTH; s++){
        h_out_ref[l].push_back(lstms[l].run(h_out_ref[l - 1][s]));
      }
    c_out_ref[l] = lstms[l].c_t;
  }
  // Results for the reference in c_out_ref[NUM_LAYERS - 1] and h_out_ref[NUM_LAYERS - 1]
  // ---------------------------------------------------------------------
  // ---------------------------------------------------------------------
  // ---------------------------------------------------------------------
  std::cout << "Running equivalent Tiramisu-FlexNLP LSTM version"<< std::endl;

  // Declare te Halide buffers (these will contain pointers to vector<> objects)
  Halide::Buffer<float*> b_input(SEQ_LENGTH); // TODO:FLEXNLP maybe use int64_t instead of float* (int64_t can contain a mem address in a 64 bits architecture)

  Halide::Buffer<float*> b_Wx(NUM_LAYERS);
  Halide::Buffer<float*> b_Wh(NUM_LAYERS);
  Halide::Buffer<float*> b_bx(NUM_LAYERS);
  Halide::Buffer<float*> b_bh(NUM_LAYERS);
  Halide::Buffer<float*> b_h_in(NUM_LAYERS);

  Halide::Buffer<float*> b_c(NUM_LAYERS);
  Halide::Buffer<float*> b_output(SEQ_LENGTH);

  // Initialize input vectors and create output vectors
  for (int s = 0; s < SEQ_LENGTH; s+=1){
    //std::vector<float>* vec_input = new std::vector<float>(x_in[s]); // Copy the reference's inputs
    b_input(s) = (float *) &x_in[s];

    // Allocate output vector
    std::vector<float>* vec_output = new std::vector<float>(h_size);
    b_output(s) = (float *) vec_output;
  }

  // Initialize the weights for each layer
  for (int l = 0; l < NUM_LAYERS; l++){
    b_Wx(l) = (float *) &Wx[l];
    b_Wh(l) = (float *) &Wh[l];
    b_bx(l) = (float *) &bx[l];
    b_bh(l) = (float *) &bh[l];

    // Initialize h_in
    std::vector<float>* vec_h_in = new std::vector<float>(h_in[l]);
    b_h_in(l) = (float *) vec_h_in;

    std::vector<float>* vec_c_in = new std::vector<float>(c_in[l]);
    b_c(l) = (float *) vec_c_in;
  }

  std::cout << "Buffers Initialized" << std::endl;

	flexnlp_test(
    b_input.raw_buffer(),
    b_Wx.raw_buffer(),
    b_Wh.raw_buffer(),
    b_bx.raw_buffer(),
    b_bh.raw_buffer(),
    b_h_in.raw_buffer(),
    b_c.raw_buffer(),
    b_output.raw_buffer()
  );

  std::cout << "H_out comparison : " << std::endl;
  for (int s = 0; s < SEQ_LENGTH; s++){
    assert(((std::vector<float>*)b_output(s))[0]  == h_out_ref[NUM_LAYERS - 1][s]);
    std::cout << "H_out[" << s <<"][0-" << FEATURE_SIZE - 1 << "] OK." << std::endl;
  }

  std::cout << "C_out comparison : " << std::endl;
  for (int l = 0; l < NUM_LAYERS; l++){
    assert(((std::vector<float>*)b_c(l))[0] == c_out_ref[l]);
    std::cout << "comparison[" << l <<"][0-" << FEATURE_SIZE - 1 << "] OK." << std::endl;
  }
  std::cout << "Correctness = 100%" << std::endl;

  return 0;
}
