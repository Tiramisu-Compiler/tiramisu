#include <Halide.h>
#include <iostream>
#include "generated_convolution.o.h"
#include "../../benchmarks.h"
#include "configure.h"
#include <tiramisu/utils.h>

int convolution_ref(Halide::Buffer<double> A, Halide::Buffer<double> F, Halide::Buffer<double> bias, Halide::Buffer<double> O)
{
    for (int b = 0; b < BATCH_SIZE; b++)
        for (int i = 0; i < INP_X-(K_X-1); i++)
            for (int j = 0; j < INP_Y-(K_Y-1); j++)
                for (int nb_f = 0; nb_f < NB_K; nb_f++){
                    O(j, i, nb_f, b) = bias(nb_f);
                    for (int c = 0; c < CHANNELS; c++)                               
                        for (int i_f = 0; i_f < K_X ; i_f++)
                            for (int j_f = 0; j_f < K_Y; j_f++)                    
                                O(j, i, nb_f, b) += F(j_f, i_f, c, nb_f)* A(j+j_f, i+i_f, c, b); 
                }                      
                

    return 0;
}

int main(int argc, char** argv)
{
    std::vector<std::chrono::duration<double, std::milli>> duration_vector_1, duration_vector_2;

    Halide::Buffer<double>  b_A(INP_Y, INP_X, CHANNELS, BATCH_SIZE), b_O(INP_Y-(K_Y-1), INP_X-(K_X-1), NB_K, BATCH_SIZE), b_O_ref(INP_Y-(K_Y-1), INP_X-(K_X-1), NB_K, BATCH_SIZE);
    Halide::Buffer<double> b_F(K_Y ,K_X ,CHANNELS, NB_K), b_bias(NB_K);

    // ---------------------------------------------------------------------
    // ---------------------------------------------------------------------
    // ---------------------------------------------------------------------

    //REFERENCE
    {
        init_4D_buffer(b_O_ref, (double) 0);
            init_buffer(b_bias, (double) 3);
            init_4D_buffer(b_A, (double) 5);
            b_A (1,2,3,4)=11;
	        init_4D_buffer(b_F, (double) 3);
        for (int i = 0; i < 1; ++i)
        {   
            
            
            auto start = std::chrono::high_resolution_clock::now();

	    	convolution_ref(b_A, b_F, b_bias, b_O_ref);

	        auto end = std::chrono::high_resolution_clock::now();
            duration_vector_1.push_back(end - start);
        }      
    }


    //TIRAMISU
    {
            init_4D_buffer(b_O, (double) 0);
            init_buffer(b_bias, (double) 3);
            init_4D_buffer(b_A, (double) 5);
            b_A (1,2,3,4)=11;
	        init_4D_buffer(b_F, (double) 3);
        for (int i = 0; i < NB_TESTS; ++i)
        {
            
            auto start = std::chrono::high_resolution_clock::now();
	    	convolution(b_A.raw_buffer(), b_F.raw_buffer(), b_bias.raw_buffer(), b_O.raw_buffer());
	        auto end = std::chrono::high_resolution_clock::now();
            duration_vector_2.push_back(end - start);
        }
    }

    print_time("performance_cpu.csv", "convolution",
	       {"Ref", "Tiramisu"},
	       {median(duration_vector_1), median(duration_vector_2)});
    
    if (CHECK_CORRECTNESS)
        compare_4D_buffers("convolution", b_O, b_O_ref, 0);

    if (PRINT_OUTPUT)
    {
        std::cout << "Tiramisu " << std::endl;
        print_buffer(b_O);
        std::cout << "Reference " << std::endl;
        print_buffer(b_O_ref);
    }


    return 0;
}
