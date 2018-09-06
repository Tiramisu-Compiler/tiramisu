#include "gaussian.h"

#include "Halide.h"
#include "halide_image_io.h"
#include "tiramisu/utils.h"
#include <cstdlib>
#include <iostream>

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

int main(int, char**)
{
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_1;
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_2;

    Halide::Buffer<uint8_t> input = Halide::Tools::load_image("../rgb.png");
    Halide::Buffer<int32_t> SIZES_b(5);
    SIZES_b(0) = input.extent(0);
    SIZES_b(1) = input.extent(1);
    SIZES_b(2) = input.extent(2);

    Halide::Buffer<float> kernelX(5);
    Halide::Buffer<float> kernelY(5);

    SIZES_b(3) = kernelX.extent(0);
    SIZES_b(4) = kernelY.extent(0);

    kernelX(0) = 1.0f; kernelX(1) = 4.0f; kernelX(2) = 6.0f; kernelX(3) = 4.0f; kernelX(4) = 1.0f;
    kernelY(0) = 1.0f/256; kernelY(1) = 4.0f/256; kernelY(2) = 6.0f/256; kernelY(3) = 4.0f/256; kernelY(4) = 1.0f/256;

    Halide::Buffer<uint8_t> output1(input.width()-8, input.height()-8, input.channels());
    Halide::Buffer<uint8_t> temp(input.width(), input.height(), input.channels());
    Halide::Buffer<uint8_t> output2(input.width()-8, input.height()-8, input.channels());

    std::cout << "Dimensions : " << std::endl;
    std::cout << "input.extent(0): " << input.extent(0) << std::endl; // Rows
    std::cout << "input.extent(1): " << input.extent(1) << std::endl; // Cols
    std::cout << "input.extent(2): " << input.extent(2) << std::endl; // Colors

    //Warm up
    pencil_gaussian(input.extent(0), input.extent(1), 1, (uint8_t *) input.raw_buffer()->host,
		    kernelX.extent(0), (float *) kernelX.raw_buffer()->host,
		    kernelY.extent(0), (float *) kernelY.raw_buffer()->host,
		    (uint8_t *) output1.raw_buffer()->host,
		    (uint8_t *) temp.raw_buffer()->host);

    // Tiramisu
    for (int i=0; i<10; i++)
    {
        auto start1 = std::chrono::high_resolution_clock::now();
        pencil_gaussian(input.extent(0), input.extent(1), 1, (uint8_t *) input.raw_buffer()->host,
   		        kernelX.extent(0), (float *) kernelX.raw_buffer()->host,
		        kernelY.extent(0), (float *) kernelY.raw_buffer()->host,
		        (uint8_t *) output1.raw_buffer()->host,
			(uint8_t *) temp.raw_buffer()->host);

        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration1 = end1 - start1;
        duration_vector_1.push_back(duration1);
    }

    std::cout << "time: " << median(duration_vector_1) << std::endl;

    return 0;
}
