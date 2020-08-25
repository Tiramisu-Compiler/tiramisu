#ifndef __FlexNLPAccelerator__
#define __FlexNLPAccelerator__

#include "Utils.h"

// Current implementation includes Behavioral Interface of
// running LSTMCell on NLP Accelerator
//  * i_size: input size (x_in size)
//  * h_size: hidden input size (h_in size)
//  * o_size: output size (h_out, c size), this parameter is refering to
//           the case of mapping LSTM to multiple accelerator by output
//           tiling
// Notes
//  * rw: 0 = read (output by reference), 1 = write,
class FlexNLPAccelerator {
 private:
  static const int usize = 16; // vector_size in hardware
  static int number_of_devices;

  int device_id; // Device's identifier for using multiple FlexNLP devices
  // Weight Buffer/SRAM
  std::vector<std::vector<float>> Wx[4]; // Wii, Wif, Wig, Wio
  std::vector<std::vector<float>> Wh[4]; // Whi, Whf, Whg, Who
  std::vector<float>  bx[4];
  std::vector<float>  bh[4];
  // Input Buffer/SRAM
  std::vector<float>  x_in;
  std::vector<float>  h_in;
  // Output Buffer/SRAM
  std::vector<float>  h_out;
  // State Buffer/SRAM
  std::vector<float>  c;

 public:
  FlexNLPAccelerator();

  int GetDeviceId();

  void ComputeLSTM();

  // For convenience, we have concat W, b for arguments
  // The concat W, b is equally split into Wii, Wig, Wif, Wio by rows
  void AccessWeightBias(
      bool rw,
      std::vector<std::vector<float>>& _Wx_concat,
      std::vector<std::vector<float>>& _Wh_concat,
      std::vector<float>&  _bx_concat,
      std::vector<float>&  _bh_concat);
  void AccessInput(
      bool rw,
      std::vector<float>&  _x_in,
      std::vector<float>&  _h_in);
  void AccessOutput(
      bool rw,
      std::vector<float>&  _h_out);
  void AccessCell(
      bool rw,
      std::vector<float>&  _c);
};

int FlexNLPAccelerator::number_of_devices = 0;

FlexNLPAccelerator::FlexNLPAccelerator(){
  this->device_id = number_of_devices;
  number_of_devices+=1;
}

int FlexNLPAccelerator::GetDeviceId(){
  return this->device_id;
}
// Compute LSTM with
//  * Input: x_in, h_in, c, Wx, Wh, bx, bh
//  * Output: h_out, c
void FlexNLPAccelerator::ComputeLSTM() {
  // The sizes must be a multiple of usize (16)
  assert (x_in.size() % usize == 0);
  assert (h_in.size() % usize == 0);
  assert (Wx[0].size() % usize == 0);

  std::vector<float> tmp_x[4];
  std::vector<float> tmp_h[4];
  std::vector<float> tmp[4];

  // XXX: Update 0703 i, g, f, o
  for (int i = 0; i < 4; i++){
    tmp_x[i] = VecAdd(MatVecMul(Wx[i], x_in), bx[i]);
    tmp_h[i] = VecAdd(MatVecMul(Wh[i], h_in), bh[i]);
    tmp[i]   = VecAdd(tmp_x[i], tmp_h[i]);
    tmp_x[i].clear();
    tmp_h[i].clear();
  }

  tmp[0] = VecSigm (tmp[0]); // i
  tmp[1] = VecSigm (tmp[1]); // f
  tmp[2] = VecTanh (tmp[2]); // g
  tmp[3] = VecSigm (tmp[3]); // o

  std::vector<float> tmp_out[2];
  tmp_out[0]  = VecMul(tmp[1], c);              // f*c
  tmp_out[1]  = VecMul(tmp[0], tmp[2]);         // i*g
  c           = VecAdd(tmp_out[0], tmp_out[1]);   // c' = f*c + i*g
  tmp_out[0]  = VecTanh(c);                     // tanh(c')
  h_out       = VecMul(tmp[3], tmp_out[0]);   // h' = o * tanh(c')
}

void FlexNLPAccelerator::AccessWeightBias(
      bool rw,
      std::vector<std::vector<float>>& _Wx_concat,
      std::vector<std::vector<float>>& _Wh_concat,
      std::vector<float>&  _bx_concat,
      std::vector<float>&  _bh_concat)
{
  if (rw == 0) { // READ
    int i_size = Wx[0][0].size();  // col size
    int h_size = Wh[0][0].size();  // col size
    int o_size = Wx[0].size();     // row size

    _Wx_concat = GetMat<float>(o_size*4, i_size);
    _Wh_concat = GetMat<float>(o_size*4, h_size);
    _bx_concat = GetVec<float>(o_size*4);
    _bh_concat = GetVec<float>(o_size*4);

    for (int g = 0; g < 4; g++) {  // for each gate
      for (int r = 0; r < o_size; r++) { // Read row by row
        _Wx_concat[o_size*g + r] = Wx[g][r];
        _Wh_concat[o_size*g + r] = Wh[g][r];
        _bx_concat[o_size*g + r] = bx[g][r];
        _bh_concat[o_size*g + r] = bh[g][r];
      }
    }
  }
  else {
    int i_size = _Wx_concat[0].size();   // col size
    int h_size = _Wh_concat[0].size();   // col size
    int o_size = _Wx_concat.size() / 4;  // row size (per gate)
    for (int g = 0; g < 4; g++) {  // for each gate
      Wx[g] = GetMat<float>(o_size, i_size);
      Wh[g] = GetMat<float>(o_size, h_size);
      bx[g] = GetVec<float>(o_size);
      bh[g] = GetVec<float>(o_size);
      for (int r = 0; r < o_size; r++) { // Write row by row
        Wx[g][r] = _Wx_concat[o_size*g + r];
        Wh[g][r] = _Wh_concat[o_size*g + r];
        bx[g][r] = _bx_concat[o_size*g + r];
        bh[g][r] = _bh_concat[o_size*g + r];
      }
    }
  }
}

void FlexNLPAccelerator::AccessInput(
    bool rw,
    std::vector<float>&  _x_in,
    std::vector<float>&  _h_in)
{
  if (rw == 0)  {
    _x_in = x_in;
    _h_in = h_in;
  }
  else {
    x_in = _x_in;
    h_in = _h_in;
  }
}
void FlexNLPAccelerator::AccessOutput(
    bool rw,
    std::vector<float>&  _h_out)
{
  if (rw == 0)  {
    _h_out = h_out;
  }
  else {
    h_out = _h_out;
  }
}

void FlexNLPAccelerator::AccessCell(
    bool rw,
    std::vector<float>&  _c)
{
  if (rw == 0)  {
    _c = c;
  }
  else {
    c = _c;
  }
}

#endif
