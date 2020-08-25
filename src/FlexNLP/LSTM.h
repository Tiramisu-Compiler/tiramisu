#ifndef __TEST_LSTM_H__
#define __TEST_LSTM_H__

#include <vector>
#include <iostream>
#include <string>
#include <math.h>
#include <iomanip>
#include <cassert>

#include "Utils.h"

// This implements Software LSTM in Cpp for verification

class LSTM {
 public:
  int isize, hsize;
  std::vector<std::vector<float>> Wx[4]; // Wii, Wif, Wig, Wio
  std::vector<std::vector<float>> Wh[4]; // Whi, Whf, Whg, Who
  std::vector<float>  bx[4];
  std::vector<float>  bh[4];
  std::vector<float>  h_t;
  std::vector<float>  c_t;

  // Default Constructor
  LSTM(){};
  LSTM(int _isize, int _hsize, bool _random=false);

  // Initialize weight
  void initialize(int _isize, int _hsize, bool _random=false);

  // Reset hidden/cell states to zeros
  void reset_state();
  void random_state();
  std::vector<float> run(const std::vector<float>& x_t);
  std::vector<std::vector<float>> run(const std::vector<std::vector<float>>& X);

  // This function returns weight in concatenated form that matched FlexNLP
  //   data arrangement pattern
  std::vector<std::vector<float>> GetWx() const;
  std::vector<std::vector<float>> GetWh() const;
  std::vector<float> Getbx() const;
  std::vector<float> Getbh() const;

};

LSTM::LSTM(int _isize, int _hsize, bool _random) {
  initialize(_isize, _hsize, _random);
  return;
}

// reset ceil state and hidden state
void LSTM::reset_state() {
  h_t = GetVec<float>(hsize);
  c_t = GetVec<float>(hsize);
}

void LSTM::random_state() {
  h_t = GetRandVec (hsize);
  c_t = GetRandVec (hsize);
}

void LSTM::initialize(int _isize, int _hsize, bool _random) {
  isize = _isize;
  hsize = _hsize;

  //assert (isize % usize == 0);
  //assert (hsize % usize == 0);

  // Initial weight biase with zero or random
  if (_random == false) {
    for (int g = 0; g < 4; g++) {
      Wx[g] = GetMat<float>(hsize, isize);
      Wh[g] = GetMat<float>(hsize, hsize);
      bx[g] = GetVec<float>(hsize);
      bh[g] = GetVec<float>(hsize);
    }
  }
  else {
    for (int g = 0; g < 4; g++) {
      Wx[g] = GetRandMat(hsize, isize);
      Wh[g] = GetRandMat(hsize, hsize);
      bx[g] = GetRandVec(hsize);
      bh[g] = GetRandVec(hsize);
    }
  }
  // Reset hidden state and cell state
  reset_state();

  return;
}
// Single timestep
std::vector<float>  LSTM::run(const std::vector<float>& x_t) {
  std::vector<float> tmp_x[4];
  std::vector<float> tmp_h[4];
  std::vector<float> tmp[4];

  // XXX: Update 0703 i, g, f, o
  for (int i = 0; i < 4; i++){
    tmp_x[i] = VecAdd(MatVecMul(Wx[i], x_t), bx[i]);
    tmp_h[i] = VecAdd(MatVecMul(Wh[i], h_t), bh[i]);
    tmp[i]   = VecAdd(tmp_x[i], tmp_h[i]);
    tmp_x[i].clear();
    tmp_h[i].clear();
  }


  tmp[0] = VecSigm (tmp[0]); // i
  tmp[1] = VecSigm (tmp[1]); // f
  tmp[2] = VecTanh (tmp[2]); // g
  tmp[3] = VecSigm (tmp[3]); // o


  std::vector<float> tmp_out[2];
  tmp_out[0]  = VecMul(tmp[1], c_t);     // f*c
  tmp_out[1]  = VecMul(tmp[0], tmp[2]);  // i*g
  c_t         = VecAdd(tmp_out[0], tmp_out[1]);    // c' = f*c + i*g
  tmp_out[0]  = VecTanh(c_t);                      // tanh(c')
  h_t         = VecMul(tmp[3], tmp_out[0]);        // h' = o * tanh(c')

  return h_t;
}

// Multiple timesteps
std::vector<std::vector<float>> LSTM::run(const std::vector<std::vector<float>>& X) {
  // X[timestep][vector]
  // rows: timestep index
  // cols: vector size  (must be multiples of usize)

  //assert (X[0].size() % usize == 0);
  int timesteps = X.size();

  // reset hidden
  reset_state();

  std::vector<std::vector<float>> H;
  for (int t = 0; t < timesteps; t++) {
    std::vector<float> _x = X[t];
    std::vector<float> _h = run(_x);

    H.push_back(_h);
  }

  return H;
}


// Get interleaving weight during concate
std::vector<std::vector<float>> LSTM::GetWx() const {
  std::vector<std::vector<float>> M = GetMat<float>(hsize*4, isize);

  //std::cout << std::setprecision(2) << std::fixed;
  //PrintMatrix(Wx[0]);
  //cout << endl;

  for (int g = 0; g < 4; g++)
    for (int r = 0; r < hsize; r++) 
      M[g*hsize + r] = Wx[g][r];
  //PrintMatrix(M);
  //cout << endl;
  return M;
}

std::vector<std::vector<float>> LSTM::GetWh() const {
  std::vector<std::vector<float>> M = GetMat<float>(hsize*4, hsize);

  for (int g = 0; g < 4; g++)
    for (int r = 0; r < hsize; r++) 
      M[g*hsize + r] = Wh[g][r];

  return M;
}

std::vector<float> LSTM::Getbx() const {
  std::vector<float> v = GetVec<float>(hsize*4);

  for (int g = 0; g < 4; g++)
    for (int r = 0; r < hsize; r++) 
      v[g*hsize + r] = bx[g][r];

  return v;
}
std::vector<float> LSTM::Getbh() const {
  std::vector<float> v = GetVec<float>(hsize*4);

  for (int g = 0; g < 4; g++)
    for (int r = 0; r < hsize; r++) 
      v[g*hsize + r] = bh[g][r];

  return v;
}

#endif