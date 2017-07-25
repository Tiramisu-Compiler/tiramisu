#include "Halide.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>

#include "wrapper_test_54.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}  // extern "C"
#endif

int main(int, char **)
{
    Halide::Buffer<uint8_t> reference_buf0(SIZE1, SIZE1, "reference_buf0");
    init_buffer(reference_buf0, (uint8_t)5);

    for (int z = 0; z < reference_buf0.channels(); z++)
    {
        for (int y = 0; y < reference_buf0.height(); y++)
        {
            for (int x = 0; x < reference_buf0.width(); x++)
            {
                if (((x - 3) * (x - 3) + (y - 3) * (y - 3)) <= 10)
                    reference_buf0(x, y, z) = 10;
            }
        }
    }

    Halide::Buffer<uint8_t> output_buf0(SIZE1, SIZE1, "output_buf0");
    init_buffer(output_buf0, (uint8_t)0);

    tiramisu_generated_code(output_buf0.raw_buffer());

    compare_buffers(std::string(TEST_NAME_STR), output_buf0, reference_buf0);

    return 0;
}
