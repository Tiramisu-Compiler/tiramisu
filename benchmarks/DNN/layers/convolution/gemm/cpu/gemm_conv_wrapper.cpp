#include "Halide.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>

#include "gemm_conv_wrapper.h"

int main(int, char **)
{
    int N = 1;
    int W = 1001;
    int H = 1000;
    int F_In = 6;
    int F_Out = 4;
    int K_W = 5;
    int K_H = 3;
    // Note that Halide indices are reversed
    Halide::Buffer<int> parameters(7);
    Halide::Buffer<float> input_padded(F_In, H + K_H - 1, W + K_W - 1, N);
    Halide::Buffer<float> input_col(K_H, K_W, F_In, H, W, N);
    Halide::Buffer<float> kernel(F_Out, K_H, K_W, F_In);
    Halide::Buffer<float> output(F_Out, H, W, N);
    Halide::Buffer<float> output_test(F_Out, H, W, N);

    parameters(0) = N;
    parameters(1) = W;
    parameters(2) = H;
    parameters(3) = F_In;
    parameters(4) = F_Out;
    parameters(5) = K_W;
    parameters(6) = K_H;

    init_buffer(input_padded, (float)0);
    init_buffer(kernel, (float)1);

    // With decimal values test might fail due to floating point arithmetic.
    for (int n = 0; n < N; n++) {
        for (int x = 0; x < W; x++) {
            for (int y = 0; y < H; y++) {
                for (int c = 0; c < F_In; c++) {
                    input_padded(c, y + (K_H - 1) / 2, x + (K_W - 1) / 2, n) = 1;
                }
            }
        }
    }

    for (int f_out = 0; f_out < F_Out; f_out++) {
        for (int k_x = 0; k_x < K_W; k_x++) {
            for (int k_y = 0; k_y < K_H; k_y++) {
                for (int f_in = 0; f_in < F_In; f_in++) {
                    kernel(f_out, k_y, k_x, f_in) = 1;
                }
            }
        }
    }

    bool test = true;
    if (test) {
        init_buffer(output_test, (float)0);
        for (int n = 0; n < N; n++) {
            for (int x = 0; x < W; x++) {
                for (int y = 0; y < H; y++) {
                    for (int k_x = 0; k_x < K_W; k_x++) {
                        for (int k_y = 0; k_y < K_H; k_y++) {
                            for (int f_in = 0; f_in < F_In; f_in++) {
                                for (int f_out = 0; f_out < F_Out; f_out++) {
                                    output_test(f_out, y, x, n) += input_padded(f_in, y + k_y, x + k_x, n) * kernel(f_out, k_y, k_x, f_in);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    std::cout << "Buffers initialized" << std::endl;

    gemm_conv(parameters.raw_buffer(), input_padded.raw_buffer(), input_col.raw_buffer(), input_col.raw_buffer(), kernel.raw_buffer(), output.raw_buffer());

    if (test) {
        compare_buffers("convs", output, output_test);
    }

    bool print = false;
    if (print) {
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                std::printf("%3.1f ", input_col(0, 0, 0, y, x, 0));
            }
            std::cout << std::endl;
        }

        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                std::printf("%3.1f ", output(0, y, x, 0));
            }
            std::cout << std::endl;
        }
    }

    return 0;
}
