// ADD:FLEXNLP (Nadir)
#include <cstdint>
#include <iostream>
#include <sstream>
#include <assert.h>
// TODO:FLEXNLP add FlexNLP header
#include "FlexNLP/tiramisu_flexnlp.h"
// TODO:FLEXNLP : Maybe we should use a shared_ptr ? to make it thread safe
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

/*
  Set of functions for setting the constants (isize, osize, hsize...)
*/
extern "C"
int tiramisu_flexnlp_set_input_size(int input_size, int device_id){
  PETop* acc = flexnlp_context->get_accelerator_by_id(device_id);
  acc->isize = input_size;
  return 0;
}

extern "C"
int tiramisu_flexnlp_set_hidden_size(int hidden_size, int device_id){
  PETop* acc = flexnlp_context->get_accelerator_by_id(device_id);
  acc->hsize = hidden_size;
  return 0;
}

extern "C"
int tiramisu_flexnlp_set_output_size(int output_size, int device_id){
  PETop* acc = flexnlp_context->get_accelerator_by_id(device_id);
  acc->osize = output_size;
  return 0;
}

extern "C"
int tiramisu_flexnlp_set_number_of_timesteps(int ntimesteps, int device_id){
  PETop* acc = flexnlp_context->get_accelerator_by_id(device_id);
  acc->ntimestep = ntimesteps;
  return 0;
}

extern "C"
int tiramisu_flexnlp_set_batch_size(int batch_size, int device_id){
  PETop* acc = flexnlp_context->get_accelerator_by_id(device_id);
  acc->nbatch = batch_size;
  return 0;
}

// Data copy functions
extern "C"
int tiramisu_flexnlp_load(void* host_data, int offset_host, int local_index, int num_elem, int device_id){
  PETop* acc = flexnlp_context->get_accelerator_by_id(device_id);
  int8_t* data = ((int8_t*) host_data) + offset_host;
  acc->runLoadStore(data, true, local_index, num_elem);
  return 0;
}

extern "C"
int tiramisu_flexnlp_store(void* host_data, int offset_host, int local_index, int num_elem, int device_id){
  PETop* acc = flexnlp_context->get_accelerator_by_id(device_id);
  int8_t* data = ((int8_t*) host_data) + offset_host;
  acc->runLoadStore(data, false, local_index, num_elem);
  return 0;
}

// Specific data copy functions
extern "C"
int tiramisu_flexnlp_load_weights(void* host_data, int offset_host, int num_elem, int device_id){
  return tiramisu_flexnlp_load(host_data, offset_host, 1, num_elem, device_id);
}

extern "C"
int tiramisu_flexnlp_load_input(void* host_data, int offset_host, int num_elem, int device_id){
  return tiramisu_flexnlp_load(host_data, offset_host, 0, num_elem, device_id);
}

extern "C"
int tiramisu_flexnlp_store_output(void* host_data, int offset_host, int num_elem, int device_id){
  return tiramisu_flexnlp_store(host_data, offset_host, 2, num_elem, device_id);
}

/**
  This function executes an LSTM layer
*/
extern "C"
int tiramisu_flexnlp_run_lstm(void* x_in, void* w_in, void* output, void* h_out, int input_size, int hidden_size, int output_size, int ntimesteps, int batch_size, int layer_number, bool load_weight, int device_id)
{
    assert(hidden_size == output_size);
    PETop* acc = flexnlp_context->get_accelerator_by_id(device_id);
    tiramisu_flexnlp_set_input_size(input_size, device_id);
    tiramisu_flexnlp_set_hidden_size(hidden_size, device_id);
    tiramisu_flexnlp_set_output_size(output_size, device_id);
    tiramisu_flexnlp_set_number_of_timesteps(ntimesteps, device_id);
    tiramisu_flexnlp_set_batch_size(batch_size, device_id);

    if (layer_number < 0)
      layer_number = 0;
    if (layer_number > 0)
      x_in = output;
    acc->runLSTM((int8_t*) x_in, ((int8_t*) w_in) + layer_number * (4 * output_size * (input_size + hidden_size)), (int8_t*) output, load_weight);

    int8_t* h_out_int8 = (int8_t*) h_out;
    // Maybe can be done with a better method

    return 0;
}

extern "C"
int tiramisu_flexnlp_run_lstm_manual(void* x_in, void* w_in, void* output, void* h_out, int input_size, int hidden_size, int output_size, int ntimesteps, int batch_size, int layer_number, int device_id)
{
    assert(hidden_size == output_size);
    PETop* acc = flexnlp_context->get_accelerator_by_id(device_id);
    tiramisu_flexnlp_set_input_size(input_size, device_id);
    tiramisu_flexnlp_set_hidden_size(hidden_size, device_id);
    tiramisu_flexnlp_set_output_size(output_size, device_id);
    tiramisu_flexnlp_set_number_of_timesteps(ntimesteps, device_id);
    tiramisu_flexnlp_set_batch_size(batch_size, device_id);

    if (layer_number < 0)
      layer_number = 0;
    if (layer_number > 0)
      x_in = output;

    int8_t* w_in_tmp = ((int8_t*) w_in) + layer_number * (4 * output_size * (input_size + hidden_size));
    assert(hidden_size == output_size && "hsize == osize for running LSTM on single accel");
    int8_t* x_h_in = (int8_t*) malloc(batch_size*(input_size+hidden_size)*sizeof(int8_t));

    int8_t* x_in_casted = (int8_t*) x_in;
    int8_t* output_casted = (int8_t*) output;
    for (int t = 0; t < ntimesteps; t++) {
      // For LSTM computation, we need to concate h_out[t-1] with x[t]
      for (int b = 0; b < batch_size; b++) {
        // Copy x
        int offset = b*(input_size+hidden_size);
        int offset_x = (t*batch_size + b)*input_size;
        int offset_h = (t*batch_size + b)*hidden_size;
        for (int i = 0; i < input_size; i++) {
          x_h_in[offset + i] = x_in_casted[offset_x + i];
        }
        // Copy h
        for (int i = 0; i < hidden_size; i++) {
          if (t == 0)
            x_h_in[offset + input_size + i] = 0;
          else
            x_h_in[offset + input_size + i] = output_casted[offset_h + i-1];
        }
      }
      // only load weight for first timestep
      int8_t* output_tmp = output_casted + t*batch_size*hidden_size;
      acc->runLSTMCellWithoutDataCopy(x_h_in, w_in_tmp, output_tmp);
      // ------ Wait For Accelerator
    }
    free(x_h_in);

    int8_t* h_out_int8 = (int8_t*) h_out;
    // Maybe can be done with a better method

    return 0;
}

/* Partition and run one LSTM a single Accelerator */
// x_in : [ntimesteps][batch_size][input_size]
// w_in : [num_layers][hidden_size/output_size][4][output_size][input_size + hidden_size]
// output : [ntimesteps][batch_size][hidden_size]
extern "C"
int tiramisu_flexnlp_run_partitioned_lstm(void* x_in, void* w_in, void* output, void* h_out, int input_size, int hidden_size, int output_size, int ntimesteps, int batch_size, int layer_number, bool load_weight, int device_id)
{

    std::cout <<hidden_size << " / " << output_size <<" = " << hidden_size/output_size << '\n';
    assert((hidden_size == ((int)(hidden_size/output_size) * output_size)) && "output_size must be a multiple of hidden_size");
    PETop* acc = flexnlp_context->use_accelerator(device_id);
    tiramisu_flexnlp_set_input_size(input_size, device_id);
    tiramisu_flexnlp_set_hidden_size(hidden_size, device_id);
    tiramisu_flexnlp_set_output_size(output_size, device_id);
    tiramisu_flexnlp_set_number_of_timesteps(ntimesteps, device_id);
    tiramisu_flexnlp_set_batch_size(batch_size, device_id);

    if (layer_number < 0)
      layer_number = 0;
    if (layer_number > 0)
      x_in = output;

    int8_t* x_h_in = (int8_t*) malloc(batch_size*(input_size+hidden_size)*sizeof(int8_t));
    int8_t* output_tmp = (int8_t*) malloc(batch_size*output_size*sizeof(int8_t));

    int8_t* x_in_casted = (int8_t*) x_in;
    int8_t* output_casted = (int8_t*) output;

    for (int t = 0; t < ntimesteps; t++) {
      // For LSTM computation, we need to concate output[t-1] with x[t]
      for (int b = 0; b < batch_size; b++) {
        // Copy x
        int offset = b*(input_size+hidden_size);
        int offset_x = (t*batch_size + b)*input_size;
        int offset_h = (t*batch_size + b)*hidden_size;
        for (int i = 0; i < input_size; i++) {
          x_h_in[offset + i] = x_in_casted[offset_x + i];
        }
        // Copy h
        for (int i = 0; i < hidden_size; i++) {
          if (t == 0)
            x_h_in[offset + input_size + i] = 0;
          else
            x_h_in[offset + input_size + i] = output_casted[offset_h + i-1];
        }
      }
      // only load weight for first timestep
      bool load_weight_tmp = (t == 0) ? load_weight : false;
      int stride_weights = (4 * output_size * (input_size + hidden_size));
      //int8_t* output_tmp = (int8_t*) output + t*batch_size*hidden_size;
      for (int o = 0; o < hidden_size/output_size; o++){
        acc->runLSTMCell(x_h_in, ((int8_t*) w_in) + (layer_number * (hidden_size / output_size) + o) * stride_weights, output_tmp, load_weight_tmp);
        // Copy data to output buffer to reformat it
        // TODO:FLEXNLP : either done like below using a temporary output buffer, OR call many times to the runLoadStore function
        //                the choice of the method will depend on the difference between the costs (latency of issuing a data copy)
        for (int b = 0; b < batch_size; b++)
          for (int oo = 0; oo < output_size; oo++)
            output_casted[(t * batch_size + b )* hidden_size + o * output_size + oo] = output_tmp[b * output_size + oo];
      }
      // ------ Wait For Accelerator
    }

    int8_t* h_out_int8 = (int8_t*) h_out;
    // Maybe can be done with a better method, used for making an h_out vector giving the last hidden vector for each layer
    // We do this because PyTorch has this output in its LSTM version

    free(x_h_in);
    flexnlp_context->release_accelerator(device_id);
    return 0;
}

/* Partition and run one LSTM on multiple Accelerators */
// x_in : [ntimesteps][batch_size][input_size]
// w_in : [num_layers][hidden_size/output_size][4][output_size][input_size + hidden_size]
// output : [ntimesteps][batch_size][hidden_size]
extern "C"
int tiramisu_flexnlp_run_partitioned_lstm_multi(void* x_in, void* w_in, void* output, void* h_out, int input_size, int hidden_size, int output_size, int ntimesteps, int batch_size, int layer_number)
{
    assert((hidden_size == ((int)(hidden_size/output_size) * output_size)) && "output_size must be a multiple of hidden_size");
    // We assume that we are using as much devices as blocks in the output
    // so ndevices_used is equal to (hidden_size/output_size), which is the number of partitions (blocks) used
    int ndevices_used = (int)(hidden_size/output_size);

    // We set the input to the output if this isn't the first layer of a multilayer LSTM
    // because in this case, the output parameter contains the input to the next layer
    if (layer_number < 0)
      layer_number = 0;
    if (layer_number > 0)
      x_in = output;

    // Get the number of devices used (given to the flexnlp_init function)
    int number_of_devices = flexnlp_context->get_number_of_devices();
    assert(ndevices_used <= number_of_devices && "There are not enough devices to perform this operation.");

    int i = 0;
    int device_index = 0;
    std::vector<PETop*> list_of_devices_to_use; // Contains the list of the Accelerators that are going to be used by the function
    std::vector<int> list_of_device_indices_to_use; // Contains the indices of the used accelerators

    // - This loop checks if the accelerators are available
    // if they are, they will be added to the list of devices to use
    // and will be marked as unavailable during the execution of this function
    // they will be released at the end of the function
    // - This loop also sets the parameters (input_size, hidden_size, output_size, ntimesteps and batch_size)
    // for each of the accelerators
    while(i < ndevices_used && device_index < number_of_devices){
      if (flexnlp_context->isAvailable(device_index)){ // Accelerator is available
        tiramisu_flexnlp_set_input_size(input_size, device_index);
        tiramisu_flexnlp_set_hidden_size(hidden_size, device_index);
        tiramisu_flexnlp_set_output_size(output_size, device_index);
        tiramisu_flexnlp_set_number_of_timesteps(ntimesteps, device_index);
        tiramisu_flexnlp_set_batch_size(batch_size, device_index);
        list_of_devices_to_use.push_back(flexnlp_context->use_accelerator(device_index));
        list_of_device_indices_to_use.push_back(device_index);
        i++;
      }
      device_index++;
    }

    // If there are not enough available devices, we exit the program with an error
    assert(i==ndevices_used && "There aren't enough available devices to use, please lower ndevices_used's value.");

    int8_t* x_h_in = (int8_t*) malloc(batch_size*(input_size+hidden_size)*sizeof(int8_t));
    int8_t* output_tmp = (int8_t*) malloc(batch_size*hidden_size*sizeof(int8_t));

    int8_t* x_in_casted = (int8_t*) x_in;
    int8_t* output_casted = (int8_t*) output;

    for (int t = 0; t < ntimesteps; t++) {
      // For LSTM computation, we need to concate output[t-1] with x[t]
      for (int b = 0; b < batch_size; b++) {
        // Copy x
        int offset = b*(input_size+hidden_size);
        int offset_x = (t*batch_size + b)*input_size;
        int offset_h = (t*batch_size + b)*hidden_size;
        for (int i = 0; i < input_size; i++) {
          x_h_in[offset + i] = x_in_casted[offset_x + i];
        }
        // Copy h
        for (int i = 0; i < hidden_size; i++) {
          if (t == 0)
            x_h_in[offset + input_size + i] = 0;
          else
            x_h_in[offset + input_size + i] = output_casted[offset_h + i-1];
        }
      }
      // only load weight for first timestep
      int stride_weights = (4 * output_size * (input_size + hidden_size));
      int stride_output_tmp = batch_size * output_size;
      int nb_output_blocks = hidden_size / output_size;

      // Make an asynchronous call for each of the accelerators
      for (int o = 0; o < nb_output_blocks; o++) // We have as many accelerators as output blocks
        list_of_devices_to_use[o]->runLSTMCell(x_h_in, ((int8_t*) w_in) + (layer_number * nb_output_blocks + o) * stride_weights, output_tmp + o * stride_output_tmp, true);


      // ------ Wait for all the Accelerators

      // Copy data to output buffer to reformat it
      // TODO:FLEXNLP : either done like below using a temporary output buffer, OR call many times to the runLoadStore function
      //                the choice of the method will depend on the difference between the costs (latency of issuing a data copy)
      for (int o = 0; o < nb_output_blocks; o++)
        for (int b = 0; b < batch_size; b++)
          for (int oo = 0; oo < output_size; oo++)
            output_casted[(t * batch_size + b) * hidden_size + o * output_size + oo] = output_tmp[(o * batch_size + b) * output_size + oo];
    }

    int8_t* h_out_int8 = (int8_t*) h_out;
    // Maybe can be done with a better method, used for making an h_out vector giving the last hidden vector for each layer
    // We do this because PyTorch has this output in its LSTM version

    // Released all of the used accelerators
    for (int device_id : list_of_device_indices_to_use)
      flexnlp_context->release_accelerator(device_id);

    free(x_h_in);
    free(output_tmp);
    return 0;
}
