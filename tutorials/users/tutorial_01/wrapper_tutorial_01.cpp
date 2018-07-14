#include "Halide.h"
#include "wrapper_tutorial_01.h"
#include "tiramisu/utils.h"
#include <cstdlib>
#include <iostream>

int main(int, char **)
{
    int N = 10;

    Halide::Buffer<int32_t> output(N);
    init_buffer(output, (int32_t)9);

    std::cout << "Array (after initialization)" << std::endl;
    print_buffer(output);

    function0(output.raw_buffer());

    std::cout << "Array after the Halide pipeline" << std::endl;
    print_buffer(output);

    Halide::Buffer<int32_t> expected(N);
    init_buffer(expected, (int32_t)7);

    compare_buffers("tutorial_01", output, expected);

    return 0;
}
