#include <Halide.h>
#include <tiramisu/tiramisu.h>
#include <iostream>
#include "generated_asum.o.h"
#include "benchmarks.h"
#include <tiramisu/utils.h>

int asum_ref(const double * x, const int * incx, double * result)
{
    result[0] = 0;
    for(int i = 0; i < N; i++)
    {
        result[0] += abs(x[i * incx[0]]);
    }

    return 0;
}

int main(int argc, char** argv)
{
    std::vector<std::chrono::duration<double, std::milli>> duration_vector_1, duration_vector_2;

    bool run_ref = false, run_tiramisu = false;

    const char* env_ref = std::getenv("RUN_REF");

    if (env_ref != NULL && env_ref[0] == '1')
        run_ref = true;

    const char* env_tiramisu = std::getenv("RUN_TIRAMISU");

    if (env_tiramisu != NULL && env_tiramisu[0] == '1')
        run_tiramisu = true;

    // ---------------------------------------------------------------------
    // ---------------------------------------------------------------------
    // ---------------------------------------------------------------------

    Halide::Buffer<int> b_incx(1);
    init_buffer(b_incx, 5);

    Halide::Buffer<double> b_x(b_incx(0) * (N - 1) + 1), b_result(1), b_result_ref(1);
    init_buffer(b_x, (double) 5);
    init_buffer(b_result, (double) 0);
    init_buffer(b_result_ref, (double) 0);

    // ---------------------------------------------------------------------
    // ---------------------------------------------------------------------
    // ---------------------------------------------------------------------

    {
        for (int i = 0; i < NB_TESTS; ++i)
        {
            auto start = std::chrono::high_resolution_clock::now();

            if (run_ref)
	    	asum_ref(b_x.data(), b_incx.data(), b_result_ref.data());

            auto end = std::chrono::high_resolution_clock::now();
            duration_vector_1.push_back(end - start);
        }
    }

    {
        for (int i = 0; i < NB_TESTS; ++i)
        {
            auto start = std::chrono::high_resolution_clock::now();

            if (run_tiramisu)
	    	asum(b_x.raw_buffer(), b_incx.raw_buffer(), b_result.raw_buffer());

            auto end = std::chrono::high_resolution_clock::now();
            duration_vector_2.push_back(end - start);
        }
    }

    print_time("performance_cpu.csv", "asum",
	       {"Ref", "Tiramisu"},
	       {median(duration_vector_1), median(duration_vector_2)});

    if (CHECK_CORRECTNESS && run_ref && run_tiramisu)
        compare_buffers("asum", b_result, b_result_ref);

    if (PRINT_OUTPUT)
    {
        std::cout << "Tiramisu " << std::endl;
        print_buffer(b_result);

        std::cout << "Reference " << std::endl;
        print_buffer(b_result_ref);
    }

    return 0;
}
