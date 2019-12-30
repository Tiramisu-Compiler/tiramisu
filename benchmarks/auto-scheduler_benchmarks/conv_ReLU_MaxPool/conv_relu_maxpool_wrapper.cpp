#include <Halide.h>
#include <iostream>
#include "generated_conv_relu_maxpool.o.h"
#include "../../benchmarks.h"
#include "configure.h"
#include <tiramisu/utils.h>

int convolution_ref(Halide::Buffer<double> A, Halide::Buffer<double> F, Halide::Buffer<double> bias, Halide::Buffer<double> O)
{   Halide::Buffer<double> b_O_conv((INP_Y-(K_Y-1)), (INP_X-(K_X-1)), NB_K, BATCH_SIZE);

    for (int b = 0; b < BATCH_SIZE; b++)
        for (int i = 0; i < INP_X-(K_X-1); i++)
            for (int j = 0; j < INP_Y-(K_Y-1); j++)
                for (int nb_f = 0; nb_f < NB_K; nb_f++){
                    b_O_conv(j, i, nb_f, b) = bias(nb_f);
                    for (int c = 0; c < CHANNELS; c++)                               
                        for (int i_f = 0; i_f < K_X ; i_f++)
                            for (int j_f = 0; j_f < K_Y; j_f++)                    
                                b_O_conv(j, i, nb_f, b) += F(j_f, i_f, c, nb_f)* A(j+j_f, i+i_f, c, b); //convolution
                }      
    for (int b = 0; b < BATCH_SIZE; b++)
        for (int i = 0; i < (INP_X-(K_X-1))/POOL_WIDTH; i++)
            for (int j = 0; j < (INP_Y-(K_Y-1))/POOL_HEIGHT; j++)
                for (int nb_f = 0; nb_f < NB_K; nb_f++){
                    O(j, i, nb_f, b) = 0;
                    for (int pool_x = 0; pool_x< POOL_WIDTH; pool_x++)
                        for (int pool_y = 0; pool_y<POOL_HEIGHT; pool_y++)
                            if (O(j, i, nb_f, b) < b_O_conv(j*POOL_HEIGHT+pool_y, i*POOL_WIDTH+pool_x, nb_f, b))
                                O(j, i, nb_f, b) = b_O_conv(j*POOL_HEIGHT+pool_y, i*POOL_WIDTH+pool_x, nb_f, b); //relu + maxpooling
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

    Halide::Buffer<double> b_A(INP_Y, INP_X, CHANNELS, BATCH_SIZE), b_O((INP_Y-(K_Y-1))/POOL_HEIGHT, (INP_X-(K_X-1))/POOL_WIDTH, NB_K, BATCH_SIZE), b_O_ref((INP_Y-(K_Y-1))/POOL_HEIGHT, (INP_X-(K_X-1))/POOL_WIDTH, NB_K, BATCH_SIZE);
    Halide::Buffer<double> b_F(K_Y ,K_X ,CHANNELS, NB_K), b_bias(NB_K);

    // ---------------------------------------------------------------------
    // ---------------------------------------------------------------------
    // ---------------------------------------------------------------------
            //init_4D_buffer(b_O, (double) 0);
            init_buffer(b_bias, (double) 3);
            rand_init_4D_buffer(b_A);
            b_A (5,5,0,0)=-11000;
	        init_4D_buffer(b_F, (double) 3);
    //REFERENCE
    {
            
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
           
        for (int i = 0; i < NB_TESTS; ++i)
        {
            
            auto start = std::chrono::high_resolution_clock::now();
	    	conv_relu_maxpool(b_A.raw_buffer(), b_F.raw_buffer(), b_bias.raw_buffer(), b_O.raw_buffer());
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
