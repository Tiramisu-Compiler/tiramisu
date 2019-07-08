#include <tiramisu/tiramisu.h>

using namespace tiramisu;

void gen(std::string name, int size, int val0, int val1)
{
    tiramisu::init(name);

    tiramisu::var i("i", 0, 100);

    tiramisu::computation S0("S0", {i}, (i != 0), tiramisu::expr((uint8_t) (val0 + val1)));

    tiramisu::codegen({S0.get_buffer()}, "build/generated_fct_test_173.o");
}

int main(int argc, char **argv)
{
    gen("func", 10, 3, 4);

    return 0;
}
