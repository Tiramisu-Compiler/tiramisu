#include <tiramisu/tiramisu.h>

using namespace tiramisu;

int main(int argc, char **argv)
{
    tiramisu::init("test_142");

    var i("i", 0, 100), j("j", 0, 200);

    input A({i, j}, p_uint32);
    computation B({i, j}, A(i, j) * 22);
    computation C({i}, B(i, 0) + 71);
    computation D({i, j}, B(i, j) + C(i));

    B.then(C, i)
     .then(D, i);

    tiramisu::codegen({A.get_buffer(), D.get_buffer()}, "build/generated_fct_test_142.o");

    return 0;
}
