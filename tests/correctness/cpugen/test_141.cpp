#include <tiramisu/tiramisu.h>

using namespace tiramisu;

int main(int argc, char **argv)
{
    tiramisu::init("test_141");

    var i("i", 0, 100), j("j", 0, 200);

    input A({i, j}, p_uint8);
    computation B({i, j}, 2 * A(i, j));

    tiramisu::codegen({A.get_buffer(), B.get_buffer()}, "generated_fct_test_141.o");

    return 0;
}
