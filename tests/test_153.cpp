#include <tiramisu/tiramisu.h>
#include <tiramisu/block.h>

using namespace tiramisu;

int main(int argc, char **argv)
{
    tiramisu::init("test_153");

    var i("i", 0, 10),
        j("j", 0, 20),
        k("k", 0, 30),
        l("l", 0, 40);
    var i0("i0"), i1("i1");

    input A({i, j, k, l}, p_uint32);
    computation B({i, j, k, l}, A(i, j, k, l) * 2);
    computation C({i, j, k, l}, B(i, j, k, l) * 3);
    // Some complicated access
    computation D({i, j, k}, C(i, j, 1, k) * 3);

    block inner_block({&B, &C});
    inner_block.tile(j, k, 5, 5);
    // Nested blocks
    block outer_block({&inner_block, &D});
    outer_block.split(i, 2, i0, i1);

    B.then(C, l)
     .then(D, i1);

    tiramisu::codegen({A.get_buffer(), D.get_buffer()}, "build/generated_fct_test_153.o");

    return 0;
}
