#include <tiramisu/tiramisu.h>

using namespace tiramisu;

void gen(std::string name, int size, int val0, int val1)
{
    tiramisu::init();

    tiramisu::function function0(name);

    tiramisu::constant N("N", tiramisu::expr((int32_t) size), p_int32, true, NULL, 0, &function0);

    tiramisu::var i("i"), j("j"), i0("i0"), j0("j0"), i1("i1"), j1("j1");
    tiramisu::expr e1 = tiramisu::expr(tiramisu::o_add, tiramisu::expr((uint8_t) val0), tiramisu::expr((uint8_t) val1));
    tiramisu::computation S0("[N]->{S0[i,j]: 0<=i<N and 0<=j<N}", e1, true, p_uint8, &function0);

    S0.tile(i, j, 2, 2, i0, j0, i1, j1);
    S0.tag_parallel_level(i0);

    tiramisu::buffer buf0("buf0", {1}, tiramisu::p_uint8, a_output, &function0);
    S0.store_in(&buf0, {});

    function0.codegen({&buf0}, "generated_fct_test_106.o");
}

int main(int argc, char **argv)
{
    gen("func", 10, 3, 4);

    return 0;
}
