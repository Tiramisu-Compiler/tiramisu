#include "Halide.h"
#include "wrapper_test_129.h"

#include <tiramisu/utils.h>

#define NN 7

int main(int, char **)
{
    Halide::Buffer<uint8_t> bufA(NN, NN);
    Halide::Buffer<uint8_t> output_buf(NN, NN);
    Halide::Buffer<uint8_t> reference_buf(NN, NN);
    init_buffer(output_buf, (uint8_t) 177);
    for (int i = 0; i < NN; i++) {
        for (int j = 0; j < NN; j++) {
            bufA(j, i) = i * 10 + j;
        }
    }
    for (int i = 0; i < NN; i++) {
        for (int j = 0; j < NN; j++) {
            reference_buf((i + j * (NN - 1)) % NN, (i + j * 2) % NN) = bufA((j + 3) % NN, i);
        }
    }

    func(bufA.raw_buffer(), output_buf.raw_buffer());
    print_buffer(output_buf);
    print_buffer(reference_buf);
    compare_buffers("test", output_buf, reference_buf);

    return 0;
}
