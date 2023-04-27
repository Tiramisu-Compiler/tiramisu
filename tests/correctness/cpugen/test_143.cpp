#include <tiramisu/tiramisu.h>
#include <tiramisu/block.h>

using namespace tiramisu;

int main(int argc, char **argv)
{
    tiramisu::init("test_143");

    var i("i", 0, 10),
        j("j", 0, 20),
        k("k", 0, 30),
        l("l", 0, 40);
    var l0("l0"), l1("l1");

    input A({i, j, k, l}, p_uint32);
    computation B({i, j, k, l}, A(i, j, k, l) * 2);
    computation C({i, j, k, l}, B(i, j, k, l) * 3);

    // Create a block and do several operations that are common.
    block b({&B, &C});
    b.parallelize(i);
    b.tile(j, k, 5, 5);
    b.vectorize(l, 4, l0, l1);

    B.then(C, l1);

    tiramisu::codegen({A.get_buffer(), C.get_buffer()}, "build/generated_fct_test_143.o");

    return 0;
}
