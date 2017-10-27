#include "wrapper_fusion.h"
#include "../benchmarks.h"

#include "Halide.h"
#include "halide_image_io.h"
#include "tiramisu/utils.h"
#include <cstdlib>
#include <iostream>

int main(int, char**)
{
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_1;
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_2;

    Halide::Buffer<uint8_t> input = Halide::Tools::load_image("./images/rgb.png");

    Halide::Buffer<uint8_t> output_ref_f(input.width(), input.height(), input.channels());
    Halide::Buffer<uint8_t> output_ref_g(input.width(), input.height(), input.channels());
    Halide::Buffer<uint8_t> output_ref_h(input.width(), input.height(), input.channels());
    Halide::Buffer<uint8_t> output_ref_k(input.width(), input.height(), input.channels());
    Halide::Buffer<uint8_t> output_tiramisu_f(input.width(), input.height(), input.channels());
    Halide::Buffer<uint8_t> output_tiramisu_g(input.width(), input.height(), input.channels());
    Halide::Buffer<uint8_t> output_tiramisu_h(input.width(), input.height(), input.channels());
    Halide::Buffer<uint8_t> output_tiramisu_k(input.width(), input.height(), input.channels());

    // Warm up
    fusion_tiramisu(input.raw_buffer(), output_tiramisu_f.raw_buffer(),
		    output_tiramisu_g.raw_buffer(), output_tiramisu_h.raw_buffer(),
		    output_tiramisu_k.raw_buffer());
    fusion_ref(input.raw_buffer(), output_ref_f.raw_buffer(), output_ref_g.raw_buffer(),
	       output_ref_h.raw_buffer(), output_ref_k.raw_buffer());

    // Tiramisu
    for (int i=0; i<NB_TESTS; i++)
    {
        auto start1 = std::chrono::high_resolution_clock::now();
        fusion_tiramisu(input.raw_buffer(), output_tiramisu_f.raw_buffer(),
			output_tiramisu_g.raw_buffer(), output_tiramisu_h.raw_buffer(),
			output_tiramisu_k.raw_buffer());
        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration1 = end1 - start1;
        duration_vector_1.push_back(duration1);
    }

    // Reference
    for (int i=0; i<NB_TESTS; i++)
    {
        auto start2 = std::chrono::high_resolution_clock::now();
        fusion_ref(input.raw_buffer(), output_ref_f.raw_buffer(), output_ref_g.raw_buffer(),
		   output_ref_h.raw_buffer(), output_ref_k.raw_buffer());
        auto end2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration2 = end2 - start2;
        duration_vector_2.push_back(duration2);
    }

    print_time("performance_CPU.csv", "fusion",
               {"Tiramisu", "Halide"},
               {median(duration_vector_1), median(duration_vector_2)});

    if (CHECK_CORRECTNESS)
	compare_buffers("Fusion",  output_ref_k, output_tiramisu_k);

    Halide::Tools::save_image(output_tiramisu_f, "./build/fusion_f_tiramisu.png");
    Halide::Tools::save_image(output_tiramisu_g, "./build/fusion_g_tiramisu.png");
    Halide::Tools::save_image(output_ref_f, "./build/fusion_f_ref.png");
    Halide::Tools::save_image(output_ref_g, "./build/fusion_g_ref.png");

    return 0;
}
