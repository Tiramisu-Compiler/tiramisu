#ifndef _PE_INT8_TOP_H_
#define _PE_INT8_TOP_H_

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>

#define PE_VECTOR_SIZE 16
#define PE_LANE_SIZE 16
// Notes:
//   * This version uses int8 as data type to simulate
//     accelerator with 8bit data movement
//   * For simplicity the code only simulates matrix-vector
//     and dataflow of Linear and LSTM, also cell state
//     is neglected
class PETop
{
 private:
  // Input, Weight, Output scratchpads
  static const int kMemSize0 = 256*64;
  static const int kMemSize1 = 256*256*8;
  static const int kMemSize2 = 256*64;

  int8_t* spad0;  // input
  int8_t* spad1;  // weight
  int8_t* spad2;  // output

 public:
  const int accel_id;
  int isize;      // input x size
  int hsize;      // input h size (only for LSTM)
  int osize;      // output h size
  int nbatch;     // number of batch
  int ntimestep;  // number of timesteps

  PETop(int _accel_id): accel_id(_accel_id)
  {
    // Allocate memory for accelerator (symbolically)
    spad0 = (int8_t*) malloc(kMemSize0*sizeof(int8_t));
    spad1 = (int8_t*) malloc(kMemSize1*sizeof(int8_t));
    spad2 = (int8_t*) malloc(kMemSize2*sizeof(int8_t));

  }
  ~PETop() {
    free(spad0);
    free(spad1);
    free(spad2);
  }
  int GetDeviceId(){
    return this->accel_id;
  }

  int GetMemSize(int local_index){
    switch(local_index){
      case 0:
        return this->kMemSize0;
        break;
      case 1:
        return this->kMemSize1;
        break;
      case 2:
        return this->kMemSize2;
        break;
    }
  }
  // NOTE:
  //  * function created for convenience of interface
  //    but not the exact data-movement in acccelerator
  void runLoadStore(int8_t* host_data, bool load_store, int local_index, int num_elem) {
    int8_t* local_data;
    if (local_index == 0) local_data = spad0;
    else if (local_index == 1) local_data = spad1;
    else if (local_index == 2) local_data = spad2;
    for (int i = 0; i < num_elem; i++) {
      if (load_store == 0) // load
        local_data[i] = host_data[i];
      else   // store
        host_data[i] = local_data[i];
    }
  }

  // x_h_in: (batch, "isize+hsize")
  // w_in:   (4*osize, isize+hsize)
  // h_out:  (batch, hsize)
  void runLSTMCell(int8_t* x_h_in, int8_t* w_in, int8_t* h_out, bool load_weight)
  {
    const int ngate = 4;
    int8_t* pe_i0 = spad0;
    int8_t* pe_w0 = spad1;
    int8_t* pe_o0 = spad2;
    int16_t* dp_accum = (int16_t*) malloc(ngate*osize*sizeof(int16_t));
    for (int i = 0; i < ngate*osize; i++)
      dp_accum[i] = (int16_t) 0;
    // ------ Accelerator Start
    runLoadStore(x_h_in, false, 0, nbatch*(isize+hsize));
    if (load_weight) {
      runLoadStore(w_in, false, 1, 4*osize*(isize+hsize));
    }
    // perform matrix vector with shift
    for (int b = 0; b < nbatch; b++) {
      for (int r = 0; r < ngate*osize; r += 1) {
        for (int c = 0; c < isize+hsize; c += 1) {
          int16_t pe_i0_int16 = pe_i0[b*isize+c];
          int16_t pe_w0_int16 = pe_w0[r*isize+c];
          dp_accum[r] = dp_accum[r] + pe_i0_int16*pe_w0_int16;
        }
      }
      // copy one batch of result (only copy 1/4)
      for (int r = 0; r < osize; r += 1) {
        dp_accum[r] = dp_accum[r] >> 6;
        if (dp_accum[r] > 127) dp_accum[r] = 127;
        if (dp_accum[r] < -127) dp_accum[r] = -127;
        int8_t out_tmp = dp_accum[r];
        // copy to output
        pe_o0[b*osize+r] = out_tmp;
      }
    }
    runLoadStore(h_out, true, 2, nbatch*osize);
    // ------ Accelerator End
    free(dp_accum);
  }

  void runLSTMCellWithoutDataCopy(int8_t* x_h_in, int8_t* w_in, int8_t* h_out)
  {
    const int ngate = 4;
    int8_t* pe_i0 = spad0;
    int8_t* pe_w0 = spad1;
    int8_t* pe_o0 = spad2;
    int16_t* dp_accum = (int16_t*) malloc(ngate*osize*sizeof(int16_t));

    // ------ Accelerator Start
    // perform matrix vector with shift
    for (int b = 0; b < nbatch; b++) {
      for (int r = 0; r < ngate*osize; r += 1) {
        for (int c = 0; c < isize+hsize; c += 1) {
          int16_t pe_i0_int16 = pe_i0[b*isize+c];
          int16_t pe_w0_int16 = pe_w0[r*isize+c];
          dp_accum[r] = dp_accum[r] + pe_i0_int16*pe_w0_int16;
        }
      }
      // copy one batch of result (only copy 1/4)
      for (int r = 0; r < osize; r += 1) {
        dp_accum[r] = dp_accum[r] >> 6;
        if (dp_accum[r] > 127) dp_accum[r] = 127;
        if (dp_accum[r] < -127) dp_accum[r] = -127;
        int8_t out_tmp = dp_accum[r];
        // copy to output
        pe_o0[b*osize+r] = out_tmp;
      }
    }
    // ------ Accelerator End
    free(dp_accum);
  }

  // x_in: (seq, batch, isize)
  // w_in: (4*osize, isize + hsize)
  // h_out: (seq, batch, osize)
  void runLSTM(int8_t* x_in, int8_t* w_in, int8_t* h_out, bool load_weight)
  {
    assert(hsize == osize && "hsize == osize for running LSTM on single accel");
    int8_t* x_h_in = (int8_t*) malloc(nbatch*(isize+hsize)*sizeof(int8_t));
    for (int t = 0; t < ntimestep; t++) {
      // For LSTM computation, we need to concate h_out[t-1] with x[t]
      for (int b = 0; b < nbatch; b++) {
        // Copy x
        int offset = b*(isize+hsize);
        int offset_x = (t*nbatch + b)*isize;
        int offset_h = (t*nbatch + b)*hsize;
        for (int i = 0; i < isize; i++) {
          x_h_in[offset + i] = x_in[offset_x + i];
        }
        // Copy h
        for (int i = 0; i < hsize; i++) {
          if (t == 0)
            x_h_in[offset + isize + i] = 0;
          else
            x_h_in[offset + isize + i] = h_out[offset_h + i-1];
        }
      }
      // only load weight for first timestep
      bool load_weight_tmp = (t == 0) ? load_weight : false;
      int8_t* h_out_tmp = h_out + t*nbatch*hsize;
      runLSTMCell(x_h_in, w_in, h_out_tmp, load_weight_tmp);
      // ------ Wait For Accelerator
    }
    free(x_h_in);
  }

  // x_in: (batch, isize)
  // w_in: (osize, isize)
  // h_out: (batch, osize)
  void runLinearCell(int8_t* x_in, int8_t* w_in, int8_t* h_out, bool load_weight)
  {
    assert(hsize == 0 && "hsize == 0 for linear layer");
    int8_t* pe_i0 = spad0;
    int8_t* pe_w0 = spad1;
    int8_t* pe_o0 = spad2;
    int16_t* dp_accum = (int16_t*) malloc(osize*sizeof(int16_t));

    // ------ Accelerator Start
    runLoadStore(x_in, false, 0, nbatch*isize);
    if (load_weight) {
      runLoadStore(w_in, false, 1, osize*isize);
    }

    // perform matrix vector with shift
    for (int b = 0; b < nbatch; b++) {
      for (int r = 0; r < osize; r += 1) {
        for (int c = 0; c < isize; c += 1) {
          int16_t pe_i0_int16 = pe_i0[b*isize+c];
          int16_t pe_w0_int16 = pe_w0[r*isize+c];
          dp_accum[r] = dp_accum[r] + pe_i0_int16*pe_w0_int16;
        }
      }
      // copy one batch of result (only copy 1/4)
      for (int r = 0; r < osize; r += 1) {
        dp_accum[r] = dp_accum[r] >> 6;
        if (dp_accum[r] > 127) dp_accum[r] = 127;
        if (dp_accum[r] < -127) dp_accum[r] = -127;
        int8_t out_tmp = dp_accum[r];
        // copy to output
        pe_o0[b*osize+r] = out_tmp;
      }
    }
    runLoadStore(h_out, true, 2, nbatch*osize);
    // ------ Accelerator End
    free(dp_accum);
  }

  // x_in: (seq, batch, isize), must be preallocated
  // h_out: (seq, batch, osize), must be preallocated
  void runLinear(int8_t* x_in, int8_t* w_in, int8_t* h_out, bool load_weight)
  {
    assert(hsize == 0 && "hsize == 0 for linear layer");

    for (int t = 0; t < ntimestep; t++) {
      // only load weight for first timestep
      bool load_weight_tmp = (t == 0) ? load_weight : false;
      int8_t* x_in_tmp = x_in + t*nbatch*isize;
      int8_t* h_out_tmp = h_out + t*nbatch*osize;
      runLinearCell(x_in, w_in, h_out_tmp, load_weight_tmp);
      // ------ Wait For Accelerator
    }
  }
};

#endif
