#include "Halide.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>
#include "mkl_cblas.h"

#include "axpy_wrapper.h"


#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}  // extern "C"
#endif


int main(int, char **)
{
    Halide::Buffer<float> a(1, "a");
    Halide::Buffer<float> x(SIZE, "x");
    Halide::Buffer<float> y_ref(SIZE, "y_ref");
    Halide::Buffer<float> y(SIZE, "y");

    init_buffer(x, (float)1);
    init_buffer(y, (float)1);
    init_buffer(y_ref, (float)1);
    init_buffer(a, (float)1);

    tiramisu_timer timer;
    timer.start();
    cblas_saxpy(SIZE, 1, x.data(), 1, y_ref.data(), 1);
    timer.stop();
    timer.print("MKL saxpy");

    timer.start();
    tiramisu_generated_code(a.raw_buffer(), x.raw_buffer(), y.raw_buffer());
    timer.stop();
    timer.print("Tiramisu saxpy");

    compare_buffers("benchmark_" + std::string(TEST_NAME_STR), y, y_ref);

    return 0;
}
