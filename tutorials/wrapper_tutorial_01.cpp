#include "Halide.h"
#include "wrapper_tutorial_01.h"
#include "coli/utils.h"
#include <cstdlib>
#include <iostream>

#define NN 10
#define MM 10

int main(int, char**)
{
    buffer_t input_buf = {0};
    input_buf.host = (unsigned char *) malloc(NN*MM*sizeof(unsigned char));
    input_buf.stride[0] = 1;
    input_buf.stride[1] = 1;
    input_buf.extent[0] = NN;
    input_buf.extent[1] = MM;
    input_buf.min[0] = 0;
    input_buf.min[1] = 0;
    input_buf.elem_size = 1;

    init_1D_buffer(&input_buf, NN*MM, 9);
    std::cout << "Array (after initialization)" << std::endl;
    print_1D_buffer(&input_buf, NN*MM);

    function0(&input_buf);

    std::cout << "Array after the Halide pipeline" << std::endl;
    print_1D_buffer(&input_buf, NN*MM);

    return 0;
}
