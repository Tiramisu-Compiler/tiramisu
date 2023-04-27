#include "Halide.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>

#include "wrapper_test_76.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}  // extern "C"
#endif

int main(int, char **)
{
    Halide::Buffer<uint16_t> reference(DSIZE), output(DSIZE);

    init_buffer(reference, (uint16_t)5);

    tiramisu_generated_code(output.raw_buffer());

    compare_buffers(TEST_ID_STR, output, reference);

    return 0;
}
