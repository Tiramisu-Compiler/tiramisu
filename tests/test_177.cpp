#include <tiramisu/tiramisu.h>

using namespace tiramisu;

int main(int argc, char** argv) {
    tiramisu::init("fib");

    input N_input("N_input", {"N_i"}, {1}, p_int32);
    constant N("N", N_input(0));

    var x("x", 0, N);

    computation fib("fib", {x}, p_int32);
    fib.set_expression(tiramisu::expr(o_select, x < 2, x, fib(x-1) + fib(x-2)));
    var r("r", 0, 1);
    computation res("res", {r}, p_int32, fib(N-1));

    res.after(fib, computation::root_dimension);

    buffer buf_f("buf_f", {2}, p_int32, a_temporary);
    fib.store_in(&buf_f, {x % 2});
    buffer buf_res("buf_res", {1}, p_int32, a_output);
    res.store_in(&buf_res, {0});

    tiramisu::codegen({N_input.get_buffer(), &buf_res}, "build/generated_fct_test_177.o", false, true);

    return 0;
}
