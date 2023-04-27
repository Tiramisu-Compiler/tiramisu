#include <tiramisu/tiramisu.h>
#include <tiramisu/block.h>

using namespace tiramisu;

int main(int argc, char **argv)
{
    // Testing interchange and uneven splitting with block API.
    tiramisu::init("test_154");

    var i("i", 0, 10),
        j("j", 0, 20),
        k("k", 0, 30),
        l("l", 0, 40);

    input A({i, j, k, l}, p_uint32);
    computation B({i, j, k, l}, A(i, j, k, l) * 2);
    computation C({i, j, k}, B(i, j, k, 2) * 3);
    computation D({j, i}, C(3, j, i) * 5);

    block b({&B, &C});
    b.interchange(i, j);
    b.split(k, 4);

    B.then(C, i)
     .then(D, j);

    tiramisu::codegen({A.get_buffer(), D.get_buffer()}, "build/generated_fct_test_154.o");

    return 0;
}
