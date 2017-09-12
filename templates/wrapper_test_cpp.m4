`#'include "Halide.h"
`#'include <tiramisu/utils.h>
`#'include <cstdlib>
`#'include <iostream>

`#'include `"wrapper_test_'TEMPLATE_TESTNUM`.h"'

`#'ifdef __cplusplus
extern "C" {
`#'endif

`#'ifdef __cplusplus
}  // extern "C"
`#'endif

int main(int, char **)
{
    // TODO: create Halide buffers for reference and output

    tiramisu_generated_code(/* TODO: provide inputs for code */);

    // TODO: do assertions. use TEST_ID_STR to name the test.
    // e.g.: compare_buffers(TEST_ID_STR, output_buffer, reference_buffer);

    return 0;
}
