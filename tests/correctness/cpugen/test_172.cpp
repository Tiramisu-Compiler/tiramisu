#include <tiramisu/tiramisu.h>

using namespace tiramisu;

void gen(std::string name, int size, int val0, int val1)
{
    tiramisu::init(name);

    tiramisu::var i("i", 0, 100);

    tiramisu::computation S0({i}, tiramisu::expr((uint8_t) (val0 + val1)));

    tiramisu::codegen({S0.get_buffer()}, "generated_fct_test_172.o");
}

int main(int argc, char **argv)
{
    gen("func", 10, 3, 4);

    return 0;
}
