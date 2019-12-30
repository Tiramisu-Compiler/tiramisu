#include <Halide.h>
#include <iostream>
#include "generated_matmul.o.h"
#include "../../benchmarks.h"
#include "configure.h"
#include <tiramisu/utils.h>


int matmul_ref(Halide::Buffer<double> A, Halide::Buffer<double> B, Halide::Buffer<double> C)
{
    for (int i = 0; i < LL; i++){
        for (int j = 0; j < NN; j++){
            C(j, i) = 0;
            for (int k = 0; k < MM; k++){   
                C(j, i) += A(k, i)* B(j, k);
            }
        }
    }                                        
                

    return 0;
}

int main(int argc, char** argv)
{
    std::vector<std::chrono::duration<double, std::milli>> duration_vector_1, duration_vector_2;
    Halide::Buffer<double>  b_A(MM, LL), b_B(NN, MM), b_C(NN, MM), b_C_ref(NN, MM);
 

    // ---------------------------------------------------------------------
    // ---------------------------------------------------------------------
    // ---------------------------------------------------------------------
    //REFERENCE
    {
        for (int i = 0; i < 1; ++i)
        {
            init_buffer(b_A, (double) 5);
            b_A (1,2) = 11;
	        init_buffer(b_B, (double) 3);

            auto start = std::chrono::high_resolution_clock::now();


	    	matmul_ref(b_A, b_B, b_C_ref);

	        auto end = std::chrono::high_resolution_clock::now();
            duration_vector_1.push_back(end - start);
        }
        
    }

    //TIRAMISU
    {
        for (int i = 0; i < NB_TESTS; ++i)
        {
            init_buffer(b_A, (double) 5);
            b_A (1,2)=11;
	        init_buffer(b_B, (double) 3);
            auto start = std::chrono::high_resolution_clock::now();

	    	matmul(b_A.raw_buffer(), b_B.raw_buffer(), b_C.raw_buffer());

	    auto end = std::chrono::high_resolution_clock::now();
            duration_vector_2.push_back(end - start);
        }
    }

    print_time("performance_cpu.csv", "matmul",
	       {"Ref", "Tiramisu"},
	       {median(duration_vector_1), median(duration_vector_2)});

    if (CHECK_CORRECTNESS)
        compare_buffers("matmul", b_C_ref, b_C);

    if (PRINT_OUTPUT)
    {
        std::cout << "Tiramisu " << std::endl;
        print_buffer(b_C);
        std::cout << "Reference " << std::endl;
        print_buffer(b_C_ref);
    }


    return 0;
}
