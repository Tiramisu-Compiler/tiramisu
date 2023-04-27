#include <tiramisu/tiramisu.h>

using namespace tiramisu;

void gen(std::string name, int size, int val0, int val1)
{
    tiramisu::init(name);

    tiramisu::constant N("N", size);

    tiramisu::var i("i", 0, N), j("j", 0, N);
    tiramisu::input A("A", {i, j}, p_uint8);
    tiramisu::input B("B", {i, j}, p_uint8);

    tiramisu::computation S0("S0", {i, j}, tiramisu::expr((uint8_t) (val0 + val1)));
    tiramisu::computation S1("S1", {i, j}, A(i,j) + B(i,j) + S0(i,j));

    S1.after(S0, computation::root);

    tiramisu::var i0("i0"), j0("j0"), i1("i1"), j1("j1");
    S0.tile(i, j, 2, 2, i0, j0, i1, j1);
    S0.parallelize(i0);
    S1.parallelize(i);

    tiramisu::buffer bufA("bufA", {size, size}, tiramisu::p_uint8, a_input);
    tiramisu::buffer bufB("bufB", {size, size}, tiramisu::p_uint8, a_input);
    tiramisu::buffer bufS0("bufS0", {size, size}, tiramisu::p_uint8, a_temporary);
    tiramisu::buffer bufS1("bufS1", {size, size}, tiramisu::p_uint8, a_output);

    A.store_in(&bufA);
    B.store_in(&bufB);
    S0.store_in(&bufS0);
    S1.store_in(&bufS1);

    tiramisu::codegen({&bufA, &bufB, &bufS1}, "generated_fct_test_128.o");
}

int main(int argc, char **argv)
{
    gen("func", 10, 3, 4);

    return 0;
}
