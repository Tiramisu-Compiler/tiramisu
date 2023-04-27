#include <tiramisu/tiramisu.h>

using namespace tiramisu;

void gen(std::string name, int size, int val0, int val1)
{
    tiramisu::init(name);

    constant c0("c0", 10);

    tiramisu::var i("i", 0, c0), j("j", 0, size);

    tiramisu::computation S0({i, j}, tiramisu::expr((uint8_t) (val0 + val1)));

    tiramisu::buffer buf0("buf0", {size, size}, tiramisu::p_uint8, a_output);
    S0.store_in(&buf0, {i ,j});

    // extracting the max value of the iterator i
    std::string constant_name = i.get_upper().get_name();
    constant *c = global::get_implicit_function()->get_invariant_by_name(constant_name);
    int val = c->get_expr().get_int32_value();
    assert(val == size);

    std::cout << std::endl << "Upper bound for i is " << val << std::endl;

    tiramisu::codegen({&buf0}, "generated_fct_test_161.o");
}

int main(int argc, char **argv)
{
    gen("func", 10, 3, 4);

    return 0;
}
