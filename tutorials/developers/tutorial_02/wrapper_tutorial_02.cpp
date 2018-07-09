#include "Halide.h"
#include "wrapper_tutorial_02.h"
#include "halide_image_io.h"
#include "tiramisu/utils.h"
#include <cstdlib>
#include <iostream>


#define NN 8 
#define MM 16
#define Val 225

int main(int, char **)
{
    
    Halide::Buffer<uint8_t> input(NN,MM);
    init_buffer(input, (uint8_t)Val);
    
    // Uncomment the following two lines if you want to view the input table
     std::cout << "Array (before Blurxy)" << std::endl;
     print_buffer(input); 

    Halide::Buffer<uint8_t> output_buf(NN,MM);

    // The blurxy takes a halide_buffer_t * as argument, when "image"
    // is passed, its buffer is actually extracted and passed
    // to the function (c++ operator overloading).

    blurxy(input.raw_buffer(), output_buf.raw_buffer());

    // Uncomment the following two lines if you want to view the output table
     std::cout << "Array (after Blurxy)" << std::endl;
     print_buffer(output_buf); 

    return 0;
}
