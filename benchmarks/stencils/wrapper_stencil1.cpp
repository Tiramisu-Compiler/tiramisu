#include "wrapper_blurxy.h"
#include "../benchmarks.h"

#include "Halide.h"
#include "halide_image_io.h"
#include "tiramisu/utils.h"
#include <cstdlib>
#include <iostream>

int main(int, char**)
{
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_1;

    Halide::Buffer<uint16_t> input = Halide::Tools::load_image("./images/rgb.png");

    for (int i=0; i<NB_TESTS; i++)
    {
        auto start1 = std::chrono::high_resolution_clock::now();
        stencil1_tiramisu(input.raw_buffer());
        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration1 = end1 - start1;
        duration_vector_1.push_back(duration1);
    }

    print_time("performance_CPU.csv", "stencil1",
               {"  Tiramisu "},
               {median(duration_vector_1)});

    return 0;
}
