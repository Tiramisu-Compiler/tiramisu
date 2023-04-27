#include <tiramisu/tiramisu.h>

using namespace tiramisu;

void gen(std::string name, int size, int val0, int val1)
{
    tiramisu::init(name);

    tiramisu::constant N("N", tiramisu::expr((int32_t) size), p_int32, true, NULL, 0);

    tiramisu::var i("i", 0, N), j("j", 0, N);
    tiramisu::var i0("i0"), j0("j0"), i1("i1"), j1("j1");

    tiramisu::computation S0({i, j}, tiramisu::expr((uint8_t) (val0 + val1)));

    S0.split(j, 2, j0, j1);
    S0.tag_unroll_level(j1, 2);

    tiramisu::buffer buf0("buf0", {size, size}, tiramisu::p_uint8, a_output);
    S0.store_in(&buf0, {i ,j});

    tiramisu::codegen({&buf0}, "generated_fct_test_148.o");
}

int main(int argc, char **argv)
{
    gen("func", 10, 3, 4);

    return 0;
}
