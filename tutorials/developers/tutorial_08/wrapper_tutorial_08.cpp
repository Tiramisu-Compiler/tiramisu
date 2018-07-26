#include "Halide.h"
#include "wrapper_tutorial_08.h"
#include "tiramisu/utils.h"
#include <cstdlib>
#include <iostream>


#define NN 10 
#define MM 20


int main(int, char **)
{
    // Index convention of Halide is the opposite of C and Tiramisu. Thus we
    // need to flip the indices when initializing and accessing Halide buffers.
    // Following buffer will have shape input[NN][MM] when passed to Tiramisu.
    Halide::Buffer<uint8_t> input(MM, NN);

    init_buffer(input, (uint8_t)3);
    
    // Uncomment the following two lines if you want to view the input table
    // std::cout << "Array (before tut_08)" << std::endl;
    // print_buffer(input); 

    Halide::Buffer<uint8_t> output(MM, NN);

    tut_08(input.raw_buffer(), output.raw_buffer());

    // Uncomment the following two lines if you want to view the output table
    // std::cout << "Array (after tut_08)" << std::endl;
    // print_buffer(output); 

    Halide::Buffer<uint8_t> expected(MM, NN);
    for (int i = 0; i < NN; i++) {
        for (int j = 0; j < MM; j++) {
            // Note the indices are flipped
            expected(j, i) = input(j, i) + i + 4;
        }
    }

    compare_buffers("tutorial 08", output, expected);

    return 0;
}
