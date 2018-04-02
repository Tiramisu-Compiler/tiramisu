#include "wrapper_fusiongpu.h"
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
    Halide::Buffer<int32_t> sizes(2);
    sizes(0) = input.extent(0);
    sizes(1) = input.extent(1);

    Halide::Buffer<uint8_t> output_ref_f(input.width(), input.height(), input.channels());
    Halide::Buffer<uint8_t> output_ref_g(input.width(), input.height(), input.channels());
    Halide::Buffer<uint8_t> output_ref_h(input.width(), input.height(), input.channels());
    Halide::Buffer<uint8_t> output_ref_k(input.width(), input.height(), input.channels());
    Halide::Buffer<uint8_t> output_tiramisu_f(input.width(), input.height(), input.channels());
    Halide::Buffer<uint8_t> output_tiramisu_g(input.width(), input.height(), input.channels());
    Halide::Buffer<uint8_t> output_tiramisu_h(input.width(), input.height(), input.channels());
    Halide::Buffer<uint8_t> output_tiramisu_k(input.width(), input.height(), input.channels());

    // Warm up
    fusiongpu_tiramisu(sizes.raw_buffer(), input.raw_buffer(), output_tiramisu_f.raw_buffer(),
		    output_tiramisu_g.raw_buffer(), output_tiramisu_h.raw_buffer(),
		    output_tiramisu_k.raw_buffer());
    fusiongpu_ref(input.raw_buffer(), output_ref_f.raw_buffer(), output_ref_g.raw_buffer(),
	       output_ref_h.raw_buffer(), output_ref_k.raw_buffer());

    // Tiramisu
    for (int i=0; i<NB_TESTS; i++)
    {
        auto start1 = std::chrono::high_resolution_clock::now();
        fusiongpu_tiramisu(sizes.raw_buffer(), input.raw_buffer(), output_tiramisu_f.raw_buffer(),
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
        fusiongpu_ref(input.raw_buffer(), output_ref_f.raw_buffer(), output_ref_g.raw_buffer(),
		   output_ref_h.raw_buffer(), output_ref_k.raw_buffer());
        auto end2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration2 = end2 - start2;
        duration_vector_2.push_back(duration2);
    }

    print_time("performance_CPU.csv", "fusiongpu",
               {"Tiramisu", "Halide"},
               {median(duration_vector_1), median(duration_vector_2)});

    Halide::Tools::save_image(output_tiramisu_h, "./build/fusiongpu_h_tiramisu.png");
    Halide::Tools::save_image(output_tiramisu_k, "./build/fusiongpu_k_tiramisu.png");
    Halide::Tools::save_image(output_ref_h, "./build/fusiongpu_h_ref.png");
    Halide::Tools::save_image(output_ref_k, "./build/fusiongpu_k_ref.png");

    if (CHECK_CORRECTNESS)
	compare_buffers("Fusion GPU",  output_ref_k, output_tiramisu_k);

    return 0;
}
