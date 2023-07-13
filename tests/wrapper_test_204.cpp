#include "Halide.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>

#include "wrapper_test_204.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}  // extern "C"
#endif

int main(int, char **)
{
    Halide::Buffer<uint8_t> reference_buf0(SIZE1, "reference_buf0");
    Halide::Buffer<uint8_t> reference_buf1(SIZE1, SIZE1, "reference_buf1");
    Halide::Buffer<uint8_t> reference_bufi(SIZE1, SIZE1, "reference_bufi");
    init_buffer(reference_buf0, (uint8_t)10);
    init_buffer(reference_buf1, (uint8_t)11);
    init_buffer(reference_bufi, (uint8_t)10);

    for (int z = 0; z < reference_bufi.channels(); z++)
    {
        for (int y = 0; y < reference_bufi.height(); y++)
        {
            for (int x = 0; x < reference_bufi.width(); x++)
            {
                reference_bufi(x, y, z) = x+y*reference_bufi.channels()*reference_bufi.height()+z*reference_bufi.channels();
            }
        }
    }
    for (int i = 0; i < SIZE0; i++)
        for (int j = 0; j < SIZE1; j++){
                reference_buf0(j) = reference_bufi(j, 0);
                for (int k = 0; k < SIZE1; k++){
                    reference_buf1(k, j) = reference_bufi(j, k);
                }
        }
	    

    Halide::Buffer<uint8_t> output_buf0(SIZE1, "output_buf0");
    init_buffer(output_buf0, (uint8_t)12);
    Halide::Buffer<uint8_t> output_buf1(SIZE1, SIZE1, "output_buf1");
    init_buffer(output_buf1, (uint8_t)11);
    Halide::Buffer<uint8_t> output_bufi(SIZE1, SIZE1, "output_bufi");
    init_buffer(output_bufi, (uint8_t)11);

    for (int z = 0; z < output_bufi.channels(); z++)
    {
        for (int y = 0; y < output_bufi.height(); y++)
        {
            for (int x = 0; x < output_bufi.width(); x++)
            {
                output_bufi(x, y, z) = x+y*output_bufi.channels()*output_bufi.height()+z*output_bufi.channels();
            }
        }
    }

    // Call the Tiramisu generated code
    tiramisu_generated_code(output_bufi.raw_buffer(), output_buf0.raw_buffer(), output_buf1.raw_buffer());
    compare_buffers(std::string(TEST_NAME_STR), output_buf0, reference_buf0);
    compare_buffers(std::string(TEST_NAME_STR), output_buf1, reference_buf1);

    return 0;
}
