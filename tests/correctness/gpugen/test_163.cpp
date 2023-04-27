#include <tiramisu/tiramisu.h>

using namespace tiramisu;

int main(int argc, char **argv)
{
    tiramisu::init("test_163");

    input params({var("v1", 0, 1)}, p_int32);
    constant shift("shift", params(0));

    input source({var("v2", 0, 100)}, p_int32);

    var i("i", 0, 10);
    computation output({i}, source(i + shift));

    tiramisu::codegen({params.get_buffer(), source.get_buffer(), output.get_buffer()}, "build/generated_fct_test_163.o");

    return 0;
}
