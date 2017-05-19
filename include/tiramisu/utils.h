#ifndef _TIRAMISU_UTILS
#define _TIRAMISU_UTILS

#include "Halide.h"
#include "tiramisu/debug.h"

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

double median(std::vector<std::chrono::duration<double, std::milli>> scores);
void print_time(const std::string &file_name, const std::string &kernel_name,
                const std::vector<std::string> &header_text,
                const std::vector<double> &time_vector);

// TODO(psuriana): init_buffer, print_buffer, copy_buffers, and compare_buffers
// assume the buffers can only be at most 3 dimensions. Make the functions
// able to handle arbitrary buffer dimension.

template<typename T>
inline void init_buffer(Halide::Buffer<T> &buf, T val)
{
    for (int z = 0; z < buf.channels(); z++)
    {
        for (int y = 0; y < buf.height(); y++)
        {
            for (int x = 0; x < buf.width(); x++)
            {
                buf(x, y, z) = val;
            }
        }
    }
}

template<typename T>
inline void print_buffer(const Halide::Buffer<T> &buf)
{
    std::string channels_size = ((buf.channels()>1)?std::to_string(buf.channels())+",":"");
    std::string heigth_size = ((buf.height()>1)?std::to_string(buf.height())+",":"");
    std::string width_size = ((buf.width()>=0)?std::to_string(buf.width()):"");

    std::cout << "Printing " << buf.name() << "[" << channels_size << heigth_size << width_size << "]: " << std::endl;

    for (int z = 0; z < buf.channels(); z++)
    {
        for (int y = 0; y < buf.height(); y++)
        {
            for (int x = 0; x < buf.width(); x++)
            {
            	if (std::is_same<T, uint8_t>::value)
            		std::cout << (int)buf(x, y, z);
            	else
	                std::cout << buf(x, y, z);

                if (x != buf.width() - 1)
                	std::cout << ", ";
            }
            std::cout << "\n";
        }
        std::cout << ((buf.height()>1)?"\n":"");
    }
    std::cout << "\n";
}

template<typename T>
inline void copy_buffer(const Halide::Buffer<T> &from, Halide::Buffer<T> &to)
{
    if ((from.dimensions() > to.dimensions()) || (from.channels() > to.channels()) ||
        (from.height() > to.height()) || (from.width() > to.width()))
    {
        tiramisu::error("'from' has bigger dimension size than 'to'. 'from' size: " +
        				std::to_string(from.dimensions()) + ", 'to' size: " +
        				std::to_string(to.dimensions()) + "\n", true);
    }

    for (int z = 0; z < from.channels(); z++)
    {
        for (int y = 0; y < from.height(); y++)
        {
            for (int x = 0; x < from.width(); x++)
            {
                to(x, y, z) = from(x, y, z);
            }
        }
    }
}

template<typename T>
inline void compare_buffers(const std::string &test, const Halide::Buffer<T> &result,
							const Halide::Buffer<T> &expected)
{
    if ((result.dimensions() != expected.dimensions()) ||
    	(result.channels() != expected.channels()) ||
        (result.height() != expected.height()) ||
        (result.width() != expected.width()))
    {
        tiramisu::error("result has different dimension size from expected\n", true);
    }

    for (int z = 0; z < result.channels(); z++) {
        for (int y = 0; y < result.height(); y++) {
            for (int x = 0; x < result.width(); x++) {
                if (result(x, y, z) != expected(x, y, z)) {
                    tiramisu::error("\033[1;31mTest " + test + " failed. Expected: " +
                    				std::to_string(expected(x, y, z)) + ", got: " +
                    				std::to_string(result(x, y, z)) + ".\033[0m\n", false);
                    return;
                }
            }
        }
    }
    tiramisu::str_dump("\033[1;32mTest " + test + " succeeded.\033[0m\n");
}

/**
 * Create an array {val1, val2, val1, val2, val1, val2, val1,
 * val2, ...}.
 */
template<typename T>
inline void init_2D_buffer_interleaving(Halide::Buffer<T> &buf, T val1, T val2)
{
    for (int y = 0; y < buf.height(); y++)
    {
        for (int x = 0; x < buf.width(); x++)
        {
            buf(x, y) = (y % 2 == 0) ? val1 : val2;
        }
    }
}

#endif
