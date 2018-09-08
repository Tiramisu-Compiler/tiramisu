#include "warpAffine.h"

#include "Halide.h"
#include "halide_image_io.h"
#include <cstdlib>
#include <iostream>
#include <stdlib.h>

double median(std::vector<std::chrono::duration<double, std::milli>> scores)
{
    double median;
    size_t size = scores.size();

    std::sort(scores.begin(), scores.end());

    if (size % 2 == 0)
    {
        median = (scores[size / 2 - 1].count() + scores[size / 2].count()) / 2;
    }
    else
    {
        median = scores[size / 2].count();
    }

    return median;
}

float random(float a, float b)
{
    return ((b - a)*((float)rand()/RAND_MAX)) + a;
}

int main(int, char**)
{
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_1;
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_2;

    Halide::Buffer<uint8_t> input = Halide::Tools::load_image("../rgb.png");

    Halide::Buffer<float> output1(input.width(), input.height());

    // Warm up
    pencil_affine_linear(input.extent(0), input.extent(1), 1,
			 (float *) input.raw_buffer()->host,
			 output1.extent(0), output1.extent(1), 1,
			 (float *) output1.raw_buffer()->host,
			 0.1, 0.1, 0.1, 0.1, 0.1, 0.1);

    // Tiramisu
    for (int i=0; i<10; i++)
    {
        auto start1 = std::chrono::high_resolution_clock::now();
        pencil_affine_linear(input.extent(0), input.extent(1), 1,
			 (float *) input.raw_buffer()->host,
			 output1.extent(0), output1.extent(1), 1,
			 (float *) output1.raw_buffer()->host,
			 0.1, 0.1, 0.1, 0.1, 0.1, 0.1);
        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration1 = end1 - start1;
        duration_vector_1.push_back(duration1);
    }

    std::cout << "time: " << median(duration_vector_1) << std::endl;

    return 0;
}
