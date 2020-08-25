
#include "LSTM.h"
#include "LSTMAcc.h"

int main ()
{
  int i_size = 16;
  int h_size = 32;

  // Software and Reference output
  LSTM lstm(i_size, h_size, true);
  lstm.random_state();
  // Software weight/bias
  auto Wx = lstm.GetWx();
  auto Wh = lstm.GetWh();
  auto bx = lstm.Getbx();
  auto bh = lstm.Getbh();
  // Software Activations
  auto x_in = GetRandVec(i_size);
  auto h_in = lstm.h_t;
  auto c_in = lstm.c_t;
  auto h_out_ref = lstm.run(x_in);
  auto c_out_ref = lstm.c_t;

  // Hardware testing, single LSTM timestep
  FlexNLPAccelerator lstm_acc;

  // 1. Write weight and bias
  lstm_acc.AccessWeightBias(true, Wx, Wh, bx, bh);
  //std::cout << std::endl;

  // 2. Write Input x(t), h(t-1)
  lstm_acc.AccessInput(true, x_in, h_in);
  // 3. Write Cell State c(t-1)
  lstm_acc.AccessCell(true, c_in);
  // 4. Compute
  lstm_acc.ComputeLSTM();
  // 5. Read hidden output
  std::vector<float> h_out;
  lstm_acc.AccessOutput(false, h_out);
  // 6. Read Cell state output
  std::vector<float> c_out;
  lstm_acc.AccessCell(false, c_out);

  // Result Checking
  assert(h_out == h_out_ref);
  assert(c_out == c_out_ref);

  std::cout << "Testing Correct" << std::endl;
  return 0;
}
