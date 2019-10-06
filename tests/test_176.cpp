#include <tiramisu/tiramisu.h>

using namespace tiramisu;

int main(int argc, char** argv) {
    tiramisu::init("reverse_order");

    static const int N = 48;

    var x("x", 0, N);

    computation fib("fib", {x}, p_int32);
    fib.set_expression(tiramisu::expr(o_select, x < 2, 0, fib(x-1) + fib(x-2)));
    buffer buf_f("buf_f", {1}, p_int32, a_temporary);
    fib.store_in(&buf_f, {0});

    tiramisu::codegen({&buf_f}, "build/generated_fct_test_176.o");
    assert(!global::get_implicit_function()->check_legality());

    return 0;
}
