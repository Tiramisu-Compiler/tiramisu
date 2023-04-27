#include "Halide.h"
#include <tiramisu/utils.h>
#include "wrapper_test_197.h"
#include "test_197_defs.h"

void test_allocation(const std::string &name)
{
    Halide::Buffer<int32_t> A(  A_size, A_size, T_size );
    Halide::Buffer<int32_t> A_ref( A_size, A_size, T_size );
    Halide::Buffer<int32_t> B( B_size, B_size, T_size );
    Halide::Buffer<int32_t> B_ref( B_size, B_size, T_size );
    for (int i = 0; i < A_size; ++i)
    {
        for (int j = 0; j < A_size; ++j)
        {
            for (int t = 0; t < T_size; ++t)
            {
                int val = std::rand() % 10 - 20;
                A(i, j, t) = val;
                A_ref(i, j, t) = 1;
            }
        }
    }

    for (int i = 0; i < B_size; ++i)
    {
        for (int j = 0; j < B_size; ++j)
        {
            for (int t = 0; t < T_size; ++t)
            {
                int val = - 20;
                B(i, j, t) = val;
                B_ref(i, j, t) = 2;
            }
        }
    }

    test_197( A.raw_buffer(), B.raw_buffer() );

    compare_buffers(name + "_A_check", A, A_ref);
    compare_buffers(name + "_B_check", B, B_ref);
}

int main(int, char **)
{
    test_allocation("test_197" );
    return 0;
}
