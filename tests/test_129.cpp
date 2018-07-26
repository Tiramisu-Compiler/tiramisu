#include <tiramisu/tiramisu.h>

using namespace tiramisu;

void gen(std::string name, int size)
{
    tiramisu::init(name);

    tiramisu::constant N("N", size);

    tiramisu::var i("i", 0, N), j("j", 0, N);
    tiramisu::input A("A", {i, j}, p_uint8);

    tiramisu::computation S0("S0", {i, j}, A(i, j));

    tiramisu::var i0("i0"), j0("j0"), i1("i1"), j1("j1");
    S0.tile(i, j, 2, 2, i0, j0, i1, j1);
    S0.parallelize(i0);

    tiramisu::buffer bufA("bufA", {size, size}, tiramisu::p_uint8, a_input);
    tiramisu::buffer bufS0("bufS0", {size, size}, tiramisu::p_uint8, a_output);

    // A complex access pattern
    A.store_in(&bufA, {i, (j + 3) % size});
    S0.store_in(&bufS0, {(i + j * 2) % size, (i + j * (size - 1)) % size});

    tiramisu::codegen({&bufA, &bufS0}, "build/generated_fct_test_129.o");
}

int main(int argc, char **argv)
{
    gen("func", 7);

    return 0;
}
