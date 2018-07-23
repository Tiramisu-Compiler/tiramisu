#include <tiramisu/tiramisu.h>

using namespace tiramisu;

void gen(std::string name, int size, int val0, int val1)
{
    tiramisu::init(name);

    tiramisu::constant N("N", tiramisu::expr((int32_t) size), p_int32, true, NULL, 0);

    tiramisu::var i("i", 0, N);

    tiramisu::computation S0({i}, tiramisu::expr((uint8_t) (val0 + val1)));

    S0.tag_parallel_level(i);

    tiramisu::buffer buf0("buf0", {size}, tiramisu::p_uint8, a_output);
    S0.store_in(&buf0, {i});

    tiramisu::codegen({&buf0}, "build/generated_fct_test_122.o");
}

int main(int argc, char **argv)
{
    gen("func", 10, 3, 4);

    return 0;
}
