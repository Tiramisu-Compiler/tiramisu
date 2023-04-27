#include "Halide.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>

#include "wrapper_test_86.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}  // extern "C"
#endif

int main(int, char **)
{
    Halide::Buffer<uint8_t> output(SIZE, SIZE, 4);
    Halide::Buffer<uint8_t> reference(SIZE, SIZE, 4);
    init_buffer(output, (uint8_t)0);

    for (int r = 0; r < 4; r++)
        for (int y = 0; y < reference.height(); y++)
            for (int x = 0; x < reference.width(); x++)
		if (x >= r)
                    reference(x, y, r) = (uint8_t) 5;
	        else
		    reference(x, y, r) = (uint8_t) 0;

    tiramisu_generated_code(output.raw_buffer());

    // print_buffer(output);
    // print_buffer(reference);
    compare_buffers(TEST_ID_STR, output, reference);

    return 0;
}
