#include "Halide.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>

#include "wrapper.h"

typedef std::chrono::duration<double,std::milli> t_duration;

int main(int argc, char *argv[])
{
    int feature_size = 512;
    // Note that indices are flipped (see tutorial 2)
    Halide::Buffer<int32_t> buf_params(1);
    Halide::Buffer<float> buf_Weights(feature_size, feature_size * 8);
    Halide::Buffer<float> buf_biases(feature_size * 4);
    Halide::Buffer<float> buf_h_prev(feature_size);
    Halide::Buffer<float> buf_c_prev(feature_size);
    Halide::Buffer<float> buf_x(feature_size);
    Halide::Buffer<float> buf_h(feature_size);
    Halide::Buffer<float> buf_c(feature_size);

    buf_params(0) = feature_size;

    lstm(buf_params.raw_buffer(),
         buf_Weights.raw_buffer(),
         buf_biases.raw_buffer(),
         buf_h_prev.raw_buffer(),
         buf_c_prev.raw_buffer(),
         buf_x.raw_buffer(),
         buf_h.raw_buffer(),
         buf_c.raw_buffer());

    std::cout << "LSTM done" << std::endl;

    return 0;
}
