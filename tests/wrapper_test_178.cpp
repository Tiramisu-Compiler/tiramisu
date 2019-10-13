#include "wrapper_test_178.h"
#include "Halide.h"

#include <tiramisu/utils.h>
#include <iostream>

int main()
{
    Halide::Buffer<int32_t> mat(width, height);
    Halide::Buffer<int32_t> res(height);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            mat(j, i) = i + j;
        }
    }

    sum_of_row(mat.raw_buffer(), res.raw_buffer());

    for (int i = 0; i < height; ++i) {
        int sum = 0;
        for (int j = 0; j < width; ++j) {
            sum += mat(j, i);
        }
        assert(res(i) == sum);
    }
    return 0;
}
