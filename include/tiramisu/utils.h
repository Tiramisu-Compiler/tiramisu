#ifndef _TIRAMISU_UTILS
#define _TIRAMISU_UTILS

#include "Halide.h"
#include "tiramisu/debug.h"

#include <chrono>
#include <iostream>
#include <fstream>
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
        ERROR("'from' has bigger dimension size than 'to'. 'from' size: " +
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
inline void compare_buffers_approximately(const std::string &test, const Halide::Buffer<T> &result,
                                          const Halide::Buffer<T> &expected, float threshold)
{
    if ((result.dimensions() != expected.dimensions()) ||
        (result.channels() != expected.channels()) ||
        (result.height() != expected.height()) ||
        (result.width() != expected.width()))
    {
        ERROR("result has different dimension size from expected\n", true);
    }

    for (int z = 0; z < result.channels(); z++) {
        for (int y = 0; y < result.height(); y++) {
            for (int x = 0; x < result.width(); x++) {
                if ((float) (result(x, y, z) - expected(x, y, z)) > threshold) {
                    ERROR("\033[1;31mTest " + test + " failed. At (" + std::to_string(x) +
                          ", " + std::to_string(y) + ", " + std::to_string(z) + "), expected: " +
                          std::to_string(expected(x, y, z)) + ", got: " +
                          std::to_string(result(x, y, z)) + ".\033[0m\n", false);
                    return;
                }
            }
        }
    }
    tiramisu::str_dump("\033[1;32mTest " + test + " succeeded.\033[0m\n");
}

template<typename T>
inline void compare_buffers_approximately(const std::string &test, const Halide::Buffer<T> &result,
                                          const Halide::Buffer<T> &expected)
{
    compare_buffers_approximately(test, result, expected, 0.1);
}

template<typename T>
inline void compare_4D_buffers(const std::string &test, const Halide::Buffer<T> &result,
                               const Halide::Buffer<T> &expected, int box)
{
    for (int n = 0; n < result.extent(3); n++) {
        for (int z = 0; z < result.extent(2); z++) {
            for (int y = 0; y < result.extent(1)-box; y++) {
                for (int x = 0; x < result.extent(0)-box; x++) {
#if 0
                    std::cout << "Comparing " << result(x, y, z, n) << " and "
                        << expected(x, y, z, n) << " at position " <<
                        "(" + std::to_string(x) + "/" + std::to_string(result.extent(0)-box) + ", " + std::to_string(y) + "/" + std::to_string(result.extent(1)-box) +
                        ", " + std::to_string(z) + "/" + std::to_string(result.extent(2)) + ", " + std::to_string(n) + "/" + std::to_string(result.extent(3)) + ")"
                        << std::endl;
#endif
                    if (result(x, y, z, n) != expected(x, y, z, n)) {
                        ERROR("\033[1;31mTest " + test + " failed. At (" + std::to_string(x) +
                              ", " + std::to_string(y) + ", " + std::to_string(z) + ", " + std::to_string(n)+ "), expected: " +
                              std::to_string(expected(x, y, z, n)) + ", got: " +
                              std::to_string(result(x, y, z, n)) + ".\033[0m\n", false);
                        return;
                    }
                }
            }
        }
    }
    tiramisu::str_dump("\033[1;32mTest " + test + " succeeded.\033[0m\n");
}

template<typename T>
inline void compare_buffers(const std::string &test, const Halide::Buffer<T> &result,
                            const Halide::Buffer<T> &expected)
{
/*    if ((result.dimensions() != expected.dimensions()) ||
        (result.channels() != expected.channels()) ||
        (result.height() != expected.height()) ||
        (result.width() != expected.width()))
    {
        ERROR("result has different dimension size from expected\n", true);
    }*/

    for (int z = 0; z < result.channels(); z++) {
        for (int y = 0; y < result.height(); y++) {
            for (int x = 0; x < result.width(); x++) {
                if (result(x, y, z) != expected(x, y, z)) {
                    ERROR("\033[1;31mTest " + test + " failed. At (" + std::to_string(x) +
                          ", " + std::to_string(y) + ", " + std::to_string(z) + "), expected: " +
                          std::to_string(expected(x, y, z)) + ", got: " +
                          std::to_string(result(x, y, z)) + ".\033[0m\n", true);
                    return;
                }
            }
        }
    }
    tiramisu::str_dump("\033[1;32mTest " + test + " succeeded.\033[0m\n");
}

template <typename T>
inline void compare_dist_buffers(const std::string &test, const Halide::Buffer<T> &expected_result) {
    // open the result file
    std::ifstream result("/tmp/" + test + "_all_ranks.txt");
    if (result.is_open()) {
        std::string line;
        for (int z = 0; z < expected_result.channels(); z++) {
            for (int y = 0; y < expected_result.height(); y++) {
                for (int x = 0; x < expected_result.width(); x++) {
                    std::getline(result, line);
                    if (line != std::to_string(expected_result(x, y, z))) {
                        ERROR("\033[1;31mTest " + test + " failed. At (" + std::to_string(x) +
                              ", " + std::to_string(y) + ", " + std::to_string(z) + "), expected: " +
                              std::to_string(expected_result(x, y, z)) + ", got: " +
                              line + ".\033[0m\n", true);
                        return;
                    }
                }
            }
        }
    } else {
        assert(false); // TODO put in appropriate error checking
    }
}

template <typename T, typename C>
inline void store_dist_results(const std::string &test, int rank, const Halide::Buffer<T> &result) {
    std::ofstream output_file;
    output_file.open("/tmp/" + test + "_rank_" + std::to_string(rank) + ".txt");
    for (int z = 0; z < result.channels(); z++) {
        for (int y = 0; y < result.height(); y++) {
            for (int x = 0; x < result.width(); x++) {
                output_file << static_cast<C>(result(x, y, z)) << std::endl;
            }
        }
    }
    output_file.flush();
    output_file.close();
}

// Combine distributed outputs into a single file. Assumes all individual output files are the same size.
void combine_dist_results(const std::string &test, std::vector<int> dims, int num_ranks);
/**
 * success: a boolean indicating whether the test succeeded.
 */
inline void print_test_results(const std::string &test, bool success)
{
    if (success == true)
        tiramisu::str_dump("\033[1;32mTest " + test + " succeeded.\033[0m\n");
    else
    ERROR("\033[1;31mTest " + test + " failed.\033[0m\n", false);
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

class tiramisu_timer
{
public:
    std::chrono::time_point<std::chrono::system_clock> start_timing, end_timing;

    void start()
    {
        start_timing = std::chrono::system_clock::now();
    }

    void stop()
    {
        end_timing = std::chrono::system_clock::now();
    }

    void print(std::string bench_name)
    {
        std::chrono::duration<double> elapsed_seconds = end_timing - start_timing;
        auto elapsed_micro_seconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed_seconds);
        std::cout << bench_name << ": " << elapsed_micro_seconds.count() << " micro-seconds\n";
    }
};

template <typename T>
class optional
{
    T value;
    bool has_value;

public:
    optional() : has_value(false) {}
    optional(T value) : value(value), has_value(true) {}

    explicit operator bool() const
    {
        return has_value;
    }

    T get()
    {
        return value;
    }
};

#endif
