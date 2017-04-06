#include <isl/set.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>

#include <tiramisu/debug.h>
#include <tiramisu/core.h>

#include <string.h>
#include <Halide.h>

// Test tag_gpu_dimensions
// Test the functionality of tagging 6 dimensions to be mapped
// to GPU. The outermost levels will be mapped to GPU blocks and
// the other 3 levels will be mapped to GPU threads.

using namespace tiramisu;

void generate_function(std::string name, int size, int val0)
{
    tiramisu::global::set_default_tiramisu_options();

    tiramisu::function function0(name);
    tiramisu::constant N("N", tiramisu::expr((int32_t) size), p_int32, true, NULL, 0, &function0);

    tiramisu::buffer buf0("buf0", 2, {size,size}, tiramisu::p_uint8, NULL, a_output, &function0);

    tiramisu::expr e0 = tiramisu::expr((uint8_t) val0);
    tiramisu::computation S0("[N]->{S0[i0,i1,i2,i3,i4,i5]: 0<=i0<N and 0<=i1<N and 0<=i2<N and 0<=i3<N and 0<=i4<N and 0<=i5<N}", e0, true, p_uint8, &function0);

    S0.set_access("[N]->{S0[i0,i1,i2,i3,i4,i5]->buf0[i0,i5]}");

    // Test if this works correctly.
    S0.tag_gpu_level(0,1,2,3,4,5);

    function0.set_arguments({&buf0});
    function0.gen_time_processor_domain();
    function0.gen_isl_ast();
    function0.gen_halide_stmt();
    function0.gen_c_code();
    function0.gen_halide_obj("build/generated_fct_test_17.o");
}

int main(int argc, char **argv)
{
    generate_function("test_tag_gpu_level", 4, 1);

    return 0;
}
