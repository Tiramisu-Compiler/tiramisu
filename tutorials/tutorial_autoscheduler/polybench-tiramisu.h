#ifndef POLYBENCH_TIRAMISU_H_
#define POLYBENCH_TIRAMISU_H_


#define NB_TESTS 10
#define CHECK_CORRECTNESS 1
#define PRINT_OUTPUT 0 


void compare_buffers_approximately(const std::string &test, const Halide::Buffer<double> &result,
                                          const Halide::Buffer<double> &expected, float threshold)
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
                if ((float) abs(result(x, y, z) - expected(x, y, z)) > threshold) {
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

void compare_buffers_approximately(const std::string &test, const Halide::Buffer<double> &result,
                                          const Halide::Buffer<double> &expected)
{
    compare_buffers_approximately(test, result, expected, 0.1);
}

/** A generalized transpose: fully inverts the dimension order. This does not move any data around in memory
* - it just permutes how it is indexed. */
int transpose(Halide::Buffer<double> buf) {
    std::vector<int> order;
    for (int i=buf.dimensions()-1; i>=0; i--)
        order.push_back(i);

    if (buf.dimensions() < 2) {
        // My, that was easy
        return 0;
    }

    std::vector<int> order_sorted = order;
    for (size_t i = 1; i < order_sorted.size(); i++) {
        for (size_t j = i; j > 0 && order_sorted[j - 1] > order_sorted[j]; j--) {
            std::swap(order_sorted[j], order_sorted[j - 1]);
            buf.transpose(j, j - 1);
        }
    }
    return 0;
}

#endif /* POLYBENCH_TIRAMISU_H_ */
