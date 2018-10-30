#include "wrapper_optical_flow.h"
#include "../benchmarks.h"

#include "Halide.h"
#include "halide_image_io.h"
#include "tiramisu/utils.h"
#include <cstdlib>
#include <iostream>
#include <stdlib.h>

int main(int, char**)
{
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_1;
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_2;

    Halide::Buffer<uint8_t> im1 = Halide::Tools::load_image("./utils/images/rgb.png");
    Halide::Buffer<uint8_t> im2 = Halide::Tools::load_image("./utils/images/rgb.png");

    Halide::Buffer<uint8_t> Ix_m(im1.width(), im1.height(), 3);
    Halide::Buffer<uint8_t> Iy_m(im1.width(), im1.height(), 3);
    Halide::Buffer<uint8_t> It_m(im1.width(), im1.height(), 3);
    Halide::Buffer<int> C1(20);
    Halide::Buffer<int> C2(20);
    Halide::Buffer<int> SIZES(2);

    SIZES(0) = im1.height();
    SIZES(1) = im1.width();

    init_buffer(C1, (int) 0);
    init_buffer(C2, (int) 0);

    // Warm up
    optical_flow_tiramisu(SIZES.raw_buffer(), im1.raw_buffer(), im2.raw_buffer(),
			  Ix_m.raw_buffer(), Iy_m.raw_buffer(), It_m.raw_buffer(),
			  C1.raw_buffer(), C2.raw_buffer());
//    optical_flow_ref(im1.raw_buffer(), output2.raw_buffer());

    // Tiramisu
    for (int i=0; i<NB_TESTS; i++)
    {
        auto start1 = std::chrono::high_resolution_clock::now();
        optical_flow_tiramisu(SIZES.raw_buffer(), im1.raw_buffer(), im2.raw_buffer(),
			  Ix_m.raw_buffer(), Iy_m.raw_buffer(), It_m.raw_buffer(),
			  C1.raw_buffer(), C2.raw_buffer());
        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration1 = end1 - start1;
        duration_vector_1.push_back(duration1);
    }

    // Reference
    for (int i=0; i<NB_TESTS; i++)
    {
        auto start2 = std::chrono::high_resolution_clock::now();
        //warp_affine_ref(im1.raw_buffer(), output2.raw_buffer());
        auto end2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration2 = end2 - start2;
        duration_vector_2.push_back(duration2);
    }

    print_time("performance_CPU.csv", "optical_flow",
               {"Tiramisu", "Halide"},
               {median(duration_vector_1), median(duration_vector_2)});

//    if (CHECK_CORRECTNESS)
//	compare_buffers_approximately("benchmark_warp_affine", It_m, It_m2);

    return 0;
}
