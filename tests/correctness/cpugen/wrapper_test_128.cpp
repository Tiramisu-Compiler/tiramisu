#include "Halide.h"
#include "wrapper_test_128.h"

#include <tiramisu/utils.h>

#define NN 10
#define MM 10

int main(int, char **)
{
    Halide::Buffer<uint8_t> reference_buf(NN, MM);
    init_buffer(reference_buf, (uint8_t)11);

    Halide::Buffer<uint8_t> bufA(NN, MM);
    init_buffer(bufA, (uint8_t)2);

    Halide::Buffer<uint8_t> bufB(NN, MM);
    init_buffer(bufB, (uint8_t)2);

    Halide::Buffer<uint8_t> output_buf(NN, MM);
    init_buffer(output_buf, (uint8_t)27); // Random value

    func(bufA.raw_buffer(), bufB.raw_buffer(), output_buf.raw_buffer());
    compare_buffers("test", output_buf, reference_buf);

    return 0;
}
