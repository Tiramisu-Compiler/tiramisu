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
    // The following is invalid store_in schedule. 
    fib.store_in(&buf_f, {0});
    buffer buf_res("buf_res", {1}, p_int32, a_output);
    res.store_in(&buf_res, {0});

    tiramisu::codegen({N_input.get_buffer(), &buf_res}, "build/generated_fct_test_176.o");
    assert(!global::get_implicit_function()->check_legality());

    return 0;
}
