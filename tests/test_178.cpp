#include <tiramisu/tiramisu.h>
#include "wrapper_test_178.h"

using namespace tiramisu;

int main(int argc, char** argv) {
    tiramisu::init("sum_of_row");

    var x("x", 0, width);
    var y("y", 0, height);

    input mat("mat", {y, x}, p_int32);

    computation sum_of_row("sum_of_row", {y, x}, p_int32);
    sum_of_row.set_expression(tiramisu::expr(o_select, x == 0, mat(y, 0), sum_of_row(y, x - 1) + mat(y, x)));
    buffer buf("buf", {height}, p_int32, a_output);
    sum_of_row.store_in(&buf, {y});

    tiramisu::codegen({mat.get_buffer(), &buf}, "build/generated_fct_test_178.o", false, true);

    return 0;
}
