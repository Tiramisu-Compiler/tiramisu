#include "Halide.h"
#include "wrapper_test_06.h"

#include <coli/utils.h>
#include <cstdlib>
#include <iostream>

#define NN 10
#define MM 10

int main(int, char**)
{
    buffer_t reference_buf = allocate_2D_buffer(NN, MM);
    init_2D_buffer_val(&reference_buf, NN, MM, 13);

    buffer_t output_buf = allocate_2D_buffer(NN, MM);

    init_2D_buffer_val(&output_buf, NN, MM, 0);
    test_c_gen(&output_buf);
    compare_2_2D_arrays("test_c_gen",
                        output_buf.host, reference_buf.host, NN, MM);

   return 0;
}
