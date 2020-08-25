// ADD:FLEXNLP (Nadir)
#include <cstdint>
#include <iostream>
#include <sstream>
#include <assert.h>
// TODO:FLEXNLP add FlexNLP header

extern "C"
int tiramisu_flexnlp_matrix_vector_multiply(int number_of_rows, int number_of_columns)
{
  std::cout << "Running flexnlp Matrix Vector multiply NROWS = " << number_of_rows << " and NCOLS = " << number_of_columns << std::endl;
  return 0;
}

#include "FlexNLP/tiramisu_flexnlp.h"
FlexNLPContext* flexnlp_context;

// Initialize the FlexNLP context by giving the number of devices
// this function creates the FlexNLPContext object which defines all of the
// FlexNLPAccelerator objects, one for each accelerator used.
extern "C"
int tiramisu_flexnlp_init(int number_of_devices){
  assert((number_of_devices > 0) && "You must use at least one FlexNLP device. flexnlp_init's parameter must be > 0");
  flexnlp_context = new FlexNLPContext(number_of_devices);
  return 0;
}

/**
  This function copies a buffer from cpu to the flexnlp device's input memory,
*/
extern "C"
int tiramisu_flexnlp_copy_input_to_device(void* x_in, void* h_in, int sequence_number, int layer_number, int device_id)
{
    FlexNLPAccelerator* acc = flexnlp_context->get_accelerator_by_id(device_id);
    // Cast
    std::vector<float>* x_in_casted;
    std::vector<float>* h_in_casted;

    if (sizeof(void*) == 8){
      x_in_casted = (std::vector<float>*)((int64_t*)x_in)[sequence_number];
      h_in_casted = (std::vector<float>*)((int64_t*)h_in)[layer_number];
    }
    else if (sizeof(void*) == 4){
      x_in_casted = (std::vector<float>*)((int32_t*)x_in)[sequence_number];
      h_in_casted = (std::vector<float>*)((int32_t*)h_in)[layer_number];
    }

    assert ((sizeof(void*) == 4 || sizeof(void*) == 8) && "Architecture is not 32bits or 64bits. Pointer size must be either 32bits (4 bytes) or 64 bits (8 bytes).");

    acc->AccessInput(1, *x_in_casted, *h_in_casted);
    return 0;
}

// TODO:FLEXNLP : add the copy functions to expr.cpp and expr.h
/**
  This function copies a buffer from cpu to the flexnlp device's output (output state) memory,
*/
extern "C"
int tiramisu_flexnlp_copy_output_to_device(void* h_out, int sequence_number, int device_id)
{
    FlexNLPAccelerator* acc = flexnlp_context->get_accelerator_by_id(device_id);
    // Cast
    std::vector<float>* h_out_casted;

    if (sizeof(void*) == 8){
      h_out_casted = (std::vector<float>*)((int64_t*)h_out)[sequence_number];
    }
    else if (sizeof(void*) == 4){
      h_out_casted = (std::vector<float>*)((int32_t*)h_out)[sequence_number];
    }

    assert ((sizeof(void*) == 4 || sizeof(void*) == 8) && "Architecture is not 32bits or 64bits. Pointer size must be either 32bits (4 bytes) or 64 bits (8 bytes).");
    acc->AccessOutput(1, *h_out_casted);
    return 0;
}

/**
  This function copies a buffer from cpu to the flexnlp device's weights memory,
*/
extern "C"
int tiramisu_flexnlp_copy_weights_to_device(void* W_x, void* W_h, void* b_x, void* b_h, int layer_number, int device_id)
{
    FlexNLPAccelerator* acc = flexnlp_context->get_accelerator_by_id(device_id);
    // Cast
    std::vector<std::vector<float>>* W_x_casted;
    std::vector<std::vector<float>>* W_h_casted;
    std::vector<float>* b_x_casted;
    std::vector<float>* b_h_casted;

    if (sizeof(void*) == 8){
      W_x_casted = (std::vector<std::vector<float>>*)((int64_t*)W_x)[layer_number];
      W_h_casted = (std::vector<std::vector<float>>*)((int64_t*)W_h)[layer_number];
      b_x_casted = (std::vector<float>*)((int64_t*)b_x)[layer_number];
      b_h_casted = (std::vector<float>*)((int64_t*)b_h)[layer_number];
    }
    else if (sizeof(void*) == 4){
      W_x_casted = (std::vector<std::vector<float>>*)((int32_t*)W_x)[layer_number];
      W_h_casted = (std::vector<std::vector<float>>*)((int32_t*)W_h)[layer_number];
      b_x_casted = (std::vector<float>*)((int32_t*)b_x)[layer_number];
      b_h_casted = (std::vector<float>*)((int32_t*)b_h)[layer_number];
    }

    assert ((sizeof(void*) == 4 || sizeof(void*) == 8) && "Architecture is not 32bits or 64bits. Pointer size must be either 32bits (4 bytes) or 64 bits (8 bytes).");
    acc->AccessWeightBias(1, *W_x_casted, *W_h_casted, *b_x_casted, *b_h_casted);
    return 0;
}

/**
  This function copies a buffer from cpu to the flexnlp device's cell state memory,
*/
extern "C"
int tiramisu_flexnlp_copy_cell_state_to_device(void* c, int layer_number, int device_id)
{
    FlexNLPAccelerator* acc = flexnlp_context->get_accelerator_by_id(device_id);
    // Cast
    std::vector<float>* c_casted;

    if (sizeof(void*) == 8){
      c_casted = (std::vector<float>*)((int64_t*)c)[layer_number];
    }
    else if (sizeof(void*) == 4){
      c_casted = (std::vector<float>*)((int32_t*)c)[layer_number];
    }

    assert ((sizeof(void*) == 4 || sizeof(void*) == 8) && "Architecture is not 32bits or 64bits. Pointer size must be either 32bits (4 bytes) or 64 bits (8 bytes).");
    acc->AccessCell(1, *c_casted);
    return 0;
}

/**
  This function copies the flexnlp device's input memory to the host,
*/
extern "C"
int tiramisu_flexnlp_copy_input_to_host(void* x_in, void* h_in, int sequence_number, int layer_number, int device_id)
{

  FlexNLPAccelerator* acc = flexnlp_context->get_accelerator_by_id(device_id);
  // Cast
  std::vector<float>* x_in_casted;
  std::vector<float>* h_in_casted;

  if (sizeof(void*) == 8){
    x_in_casted = (std::vector<float>*)((int64_t*)x_in)[sequence_number];
    h_in_casted = (std::vector<float>*)((int64_t*)h_in)[layer_number];
  }
  else if (sizeof(void*) == 4){
    x_in_casted = (std::vector<float>*)((int32_t*)x_in)[sequence_number];
    h_in_casted = (std::vector<float>*)((int32_t*)h_in)[layer_number];
  }

  assert ((sizeof(void*) == 4 || sizeof(void*) == 8) && "Architecture is not 32bits or 64bits. Pointer size must be either 32bits (4 bytes) or 64 bits (8 bytes).");
  acc->AccessInput(0, *x_in_casted, *h_in_casted);
  return 0;
}

/**
  This function copies the flexnlp device's output (output state) memory to the host.
*/
extern "C"
int tiramisu_flexnlp_copy_output_to_host(void* h_out, int sequence_number, int device_id)
{
  FlexNLPAccelerator* acc = flexnlp_context->get_accelerator_by_id(device_id);
  // Cast
  std::vector<float>* h_out_casted;

  if (sizeof(void*) == 8){
    h_out_casted = (std::vector<float>*)((int64_t*)h_out)[sequence_number];
  }
  else if (sizeof(void*) == 4){
    h_out_casted = (std::vector<float>*)((int32_t*)h_out)[sequence_number];
  }

  assert ((sizeof(void*) == 4 || sizeof(void*) == 8) && "Architecture is not 32bits or 64bits. Pointer size must be either 32bits (4 bytes) or 64 bits (8 bytes).");
  acc->AccessOutput(0, *h_out_casted);
  return 0;
}

/**
  This function copies the flexnlp device's weights memory to the host.
*/
extern "C"
int tiramisu_flexnlp_copy_weights_to_host(void* W_x, void* W_h, void* b_x, void* b_h, int layer_number, int device_id)
{
  FlexNLPAccelerator* acc = flexnlp_context->get_accelerator_by_id(device_id);
  // Cast
  std::vector<std::vector<float>>* W_x_casted;
  std::vector<std::vector<float>>* W_h_casted;
  std::vector<float>* b_x_casted;
  std::vector<float>* b_h_casted;

  if (sizeof(void*) == 8){
    W_x_casted = (std::vector<std::vector<float>>*)((int64_t*)W_x)[layer_number];
    W_h_casted = (std::vector<std::vector<float>>*)((int64_t*)W_h)[layer_number];
    b_x_casted = (std::vector<float>*)((int64_t*)b_x)[layer_number];
    b_h_casted = (std::vector<float>*)((int64_t*)b_h)[layer_number];
  }
  else if (sizeof(void*) == 4){
    W_x_casted = (std::vector<std::vector<float>>*)((int32_t*)W_x)[layer_number];
    W_h_casted = (std::vector<std::vector<float>>*)((int32_t*)W_h)[layer_number];
    b_x_casted = (std::vector<float>*)((int32_t*)b_x)[layer_number];
    b_h_casted = (std::vector<float>*)((int32_t*)b_h)[layer_number];
  }

  assert ((sizeof(void*) == 4 || sizeof(void*) == 8) && "Architecture is not 32bits or 64bits. Pointer size must be either 32bits (4 bytes) or 64 bits (8 bytes).");
  acc->AccessWeightBias(0, *W_x_casted, *W_h_casted, *b_x_casted, *b_h_casted);
  return 0;
}

/**
  This function copies the flexnlp device's cell state memory to the host.
*/
extern "C"
int tiramisu_flexnlp_copy_cell_state_to_host(void* c, int layer_number, int device_id)
{
  FlexNLPAccelerator* acc = flexnlp_context->get_accelerator_by_id(device_id);
  // Cast
  std::vector<float>* c_casted;

  if (sizeof(void*) == 8){
    c_casted = (std::vector<float>*)((int64_t*)c)[layer_number];
  }
  else if (sizeof(void*) == 4){
    c_casted = (std::vector<float>*)((int32_t*)c)[layer_number];
  }

  assert ((sizeof(void*) == 4 || sizeof(void*) == 8) && "Architecture is not 32bits or 64bits. Pointer size must be either 32bits (4 bytes) or 64 bits (8 bytes).");
  acc->AccessCell(0, *c_casted);
  return 0;
}

extern "C"
int tiramisu_flexnlp_lstm_cell(void* W_x, void* W_h, void* b_x, void* b_h, void* x, void* h_in, void* h_out, void* c_in, int device_id)
{
  FlexNLPAccelerator* acc = flexnlp_context->get_accelerator_by_id(device_id);
  acc->ComputeLSTM();
  return 0;
}

extern "C"
int tiramisu_copy_vector(void* to, void* from, int offset_to, int offset_from)
{
    assert ((sizeof(void*) == 4 || sizeof(void*) == 8) && "Architecture is not 32bits or 64bits. Pointer size must be either 32bits (4 bytes) or 64 bits (8 bytes).");
    // Cast
    std::vector<float>* to_casted;
    std::vector<float>* from_casted;

    if (sizeof(void*) == 8){
      to_casted = (std::vector<float>*)((int64_t*)to)[offset_to];
      from_casted = (std::vector<float>*)((int64_t*)from)[offset_from];
    }
    else if (sizeof(void*) == 4){
      to_casted = (std::vector<float>*)((int32_t*)to)[offset_to];
      from_casted = (std::vector<float>*)((int32_t*)from)[offset_from];
    }
    assert((*to_casted).size() == (*from_casted).size());
    for (int i = 0; i < (*to_casted).size(); i++)
      (*to_casted)[i] = (*from_casted)[i];

    return 0;
}
