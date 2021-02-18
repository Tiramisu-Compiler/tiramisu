#include <tiramisu/tiramisu.h>
#include <vector>
#include "configure.h"

using namespace tiramisu;

int main()
{
    init("add_relu_inplace_32_64_16_block");

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------
    var i("i", 0, N*N*FIn*BATCH_SIZE);
    input c_x("c_x", {i}, p_float32);
    input c_y("c_y", {i}, p_float32);

    computation add_relu(
        "add_relu",
        {i},
        expr(o_max, 0.f, c_x(i) + c_y(i))
    );

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    add_relu.store_in(c_y.get_buffer(), {i});

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    codegen({
        c_x.get_buffer(),
        c_y.get_buffer()
    }, "add_relu_inplace_32_64_16_tiramisu.o");

    return 0;
}
