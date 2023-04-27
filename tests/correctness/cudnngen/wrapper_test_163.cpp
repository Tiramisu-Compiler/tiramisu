#include "Halide.h"
#include "wrapper_test_163.h"

#include <tiramisu/utils.h>

int main(int, char **)
{
    int shift = 5;
    Halide::Buffer<int32_t> params(1);
    params(0) = shift;

    Halide::Buffer<int32_t> output(10);
    Halide::Buffer<int32_t> source(100);
    Halide::Buffer<int32_t> reference(10);
    for (int i = 0; i < 100; i++) source(i) = i;
    for (int i = 0; i < 10; i++) reference(i) = i + shift;
    init_buffer(output, 17);

    test_163(params.raw_buffer(), source.raw_buffer(), output.raw_buffer());
    compare_buffers("test_163", output, reference);

    return 0;
}
