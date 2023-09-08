#include <tiramisu/tiramisu.h>

using namespace tiramisu;

void gen(std::string name, int size, int val0, int val1)
{
    tiramisu::init(name);

    tiramisu::var i("i", 0, size), j("j", 0, size);
    tiramisu::var i0("i0"), j0("j0"), i1("i1"), j1("j1");
    tiramisu::var i00("i00"), j00("j00"), i01("i01"), j01("j01");

    tiramisu::computation S0({i, j}, tiramisu::expr((uint8_t) (val0 + val1)));

    S0.tile(i, j, 2, 2, i0, j0, i1, j1);
    S0.tile(i0, j0, 2, 2, i00, j00, i01, j01);
    S0.unroll(i1, 2);
    S0.vectorize(j1, 2);

    tiramisu::buffer buf0("buf0", {size, size}, tiramisu::p_uint8, a_output);
    S0.store_in(&buf0, {i ,j});

    tiramisu::codegen({&buf0}, "build/generated_fct_test_159.o");
}

int main(int argc, char **argv)
{
    gen("func", 10, 3, 4);

    return 0;
}
