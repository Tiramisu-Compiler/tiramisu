#include "Halide.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>

#include "wrapper_test_205.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}  // extern "C"
#endif
void init_array(Halide::Buffer<double> reference_buf1, int n){

    int i, j;

    for (i = 0; i < n; i++)
    {
        for (j = 0; j <= i; j++)
            reference_buf1(j, i) = (double)(-j % n) / n + 1;
        for (j = i+1; j < n; j++) {
            reference_buf1(j, i) = 0;
        }
        reference_buf1(i, i) = 1;
    }
    Halide::Buffer<double> reference_buf2(SIZE0, SIZE0, "reference_buf2");
    init_buffer(reference_buf2, (double)5);
    
    /* Make the matrix positive semi-definite. */
    /* not necessary for LU, but using same code as cholesky */
    int r,s,t;
    for (r = 0; r < n; ++r)
        for (s = 0; s < n; ++s)
            reference_buf2(s, r) = 0;
    for (t = 0; t < n; ++t)
        for (r = 0; r < n; ++r)
            for (s = 0; s < n; ++s)
                reference_buf2(s, r) += reference_buf1(t, r) * reference_buf1(t, s);
        for (r = 0; r < n; ++r)
            for (s = 0; s < n; ++s)
                reference_buf1(s, r) = reference_buf2(s, r);
}
int main(int, char **)
{
    Halide::Buffer<double> reference_buf1(SIZE0, SIZE0, "reference_buf1");
    init_array(reference_buf1, SIZE0);
    
    // LU c code
    for (int i = 0; i < SIZE0; i++) {
        for (int j = 0; j <i; j++) {
            for (int k = 0; k < j; k++) {
                reference_buf1(j, i) -= reference_buf1(k, i) * reference_buf1(j, k);
            }
            reference_buf1(j, i) /= reference_buf1(j, j);
        }
        for (int j = i; j < SIZE0; j++) {
            for (int k = 0; k < i; k++) {
                reference_buf1(j, i) -= reference_buf1(k, i) * reference_buf1(j, k);
            }
        }
    }
	print_buffer(reference_buf1);

    Halide::Buffer<double> output_buf1(SIZE0, SIZE0, "output_buf1");
    init_array(output_buf1, SIZE0);

    // Call the Tiramisu generated code
    tiramisu_generated_code(output_buf1.raw_buffer());
    print_buffer(output_buf1);
    compare_buffers(std::string(TEST_NAME_STR), output_buf1, reference_buf1);

    return 0;
}
