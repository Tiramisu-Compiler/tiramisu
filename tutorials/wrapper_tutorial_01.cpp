#include "Halide.h"
#include "wrapper_tutorial_01.h"
#include "tiramisu/utils.h"
#include <cstdlib>
#include <iostream>

#define NN 10
#define MM 10

int main(int, char**)
{
    buffer_t input_buf = allocate_2D_buffer(NN, MM);
    init_2D_buffer_val(&input_buf, NN, MM, 9);

    std::cout << "Array (after initialization)" << std::endl;
    print_2D_buffer(&input_buf, NN, MM);

    Halide::Buffer<uint8_t> halide_input_buf(input_buf);

    function0(halide_input_buf.raw_buffer());

    std::cout << "Array after the Halide pipeline" << std::endl;
    print_1D_array(halide_input_buf.data(), NN*MM);

    return 0;
}
