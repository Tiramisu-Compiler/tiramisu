#include "wrapper_test_177.h"
#include "Halide.h"

#include <tiramisu/utils.h>
#include <iostream>

int main()
{
    Halide::Buffer<int32_t> N(1);
    Halide::Buffer<int32_t> res(1);

    N(0) = 32;

    fib(N.raw_buffer(), res.raw_buffer());

    int f0 = 0, f1 = 1;
    for (int i = 2; i < N(0); ++i) {
        int f2 = f1 + f0;
        f0 = f1;
        f1 = f2;
    }
    std::cout << res(0) << std::endl;
    std::cout << f1 << std::endl;
    assert(res(0) == f1);
    
    return 0;
}
