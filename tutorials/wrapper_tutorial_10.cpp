#include "Halide.h"
#include "wrapper_tutorial_10.h"
#include "tiramisu/utils.h"
#include <cstdlib>
#include <iostream>

#define NN 10
#define MM 10

int main(int, char **)
{
    Halide::Buffer<uint8_t> output(NN, MM);
    init_buffer(output, (uint8_t)9);

    Halide::Buffer<uint8_t> expected(NN, MM);
    init_buffer(expected, (uint8_t)7);

    function0(output.raw_buffer());

    std::cout << "Array after the Halide pipeline" << std::endl;
    print_buffer(output);

   // compare_buffers("tutorial_10", output, expected);

    return 0;
}
