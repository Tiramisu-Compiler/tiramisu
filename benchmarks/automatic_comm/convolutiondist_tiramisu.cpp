#include <tiramisu/tiramisu.h>

#include <Halide.h>
#include "wrapper_convolutiondist.h"

using namespace tiramisu;

int main(int argc, char **argv)
{
    tiramisu::init("convolutiondist_tiramisu");

    function *f = global::get_implicit_function();
    f->add_context_constraints("[ROWS]->{: ROWS = "+std::to_string(_ROWS)+"}");

    constant SIZE0("COLS", _COLS);
    constant SIZE1("ROWS", _ROWS);
    constant SIZE2("CHANNELS", _CHANNELS);

    int kernel_extent_1 = 3;
    int kernel_extent_0 = 3;

    var i2("i2", 0, SIZE2), i1("i1", 0, SIZE0), i0("i0", 0, SIZE1);
    var k0("k0", 0, kernel_extent_0), k1("k1", 0, kernel_extent_1);
    var c("c", 0, SIZE2), y("y", 0, SIZE0-8), x("x", 0, SIZE1-8);

    input in("in", {i0, i1, i2}, p_int32);
    input kernel("kernel", {k1, k0}, p_float32);

    computation conv("conv", {x, y, c}, cast( p_int32, (cast(p_float32, cast(p_float32, in(x, y, c))*kernel(0, 0)) +
						       cast(p_float32, cast(p_float32, in(x+1, y, c))*kernel(0, 1)) +
						       cast(p_float32, cast(p_float32, in(x+2, y, c))*kernel(0, 2)) +
						       cast(p_float32, cast(p_float32, in(x, y+1, c))*kernel(1, 0)) +
						       cast(p_float32, cast(p_float32, in(x+1, y+1, c))*kernel(1, 1)) +
						       cast(p_float32, cast(p_float32, in(x+2, y+1, c))*kernel(1, 2)) +
						       cast(p_float32, cast(p_float32, in(x, y+2, c))*kernel(2, 0)) +
						       cast(p_float32, cast(p_float32, in(x+1, y+2, c))*kernel(2, 1)) +
						       cast(p_float32, cast(p_float32, in(x+2, y+2, c))*kernel(2, 2))
						       )));


    var i01("i01"), i02("i02");
    in.split(i0, _ROWS/_NODES, i01, i02);
    in.tag_distribute_level(i01);
    in.drop_rank_iter(i01);

    conv.split(x, _ROWS/_NODES, i01, i02);
    conv.tag_distribute_level(i01);
    conv.drop_rank_iter(i01);

    buffer buff_input("buff_input", {_ROWS / _NODES, _COLS, _CHANNELS}, p_int32, a_input);
    buffer buff_kernel("buff_kernel", {kernel_extent_1, kernel_extent_0}, p_float32, a_input);
    buffer buff_convolution("buff_convolution", {_ROWS / _NODES, _COLS-8, _CHANNELS}, p_int32, a_output);

    in.store_in(&buff_input);
    conv.store_in(&buff_convolution);
    kernel.store_in(&buff_kernel);

    conv.gen_communication();

    tiramisu::codegen({&buff_input, &buff_kernel, &buff_convolution} ,"build/generated_fct_convolutiondist.o");

    return 0;
}
