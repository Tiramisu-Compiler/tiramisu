
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

using namespace tiramisu;

void generate_function_1(std::string name, int size)
{
    global::set_default_tiramisu_options();

    function function0(name);

    var i("i"), j("j");

    constant N("N", tiramisu::expr((int32_t) size), p_int32, true, NULL, 0, &function0);

    computation S0("[N]->{S0[i,j]: 0<=i<N and 0<=j<N}", expr((int32_t) 15), true, p_int32,
                   &function0);

    computation S1("[N]->{S1[i,j]: 0<=i<N and 0<=j<N}", S0(i, j) * expr((int32_t) 2), true, p_int32,
                   &function0);

    computation S2("[N]->{S2[i,j]: 0<=i<N and 0<=j<N}", S1(i, j) / expr((int32_t) 5), true, p_int32,
                   &function0);

    S2.after(S0, i);

    S1.between(S0, 1, S2, computation::root_dimension);

    S0.allocate_and_map_buffer_automatically();
    S1.allocate_and_map_buffer_automatically();
    S2.allocate_and_map_buffer_automatically(a_output);

    function0.set_arguments({S2.get_automatically_allocated_buffer()});
    function0.gen_time_space_domain();
    function0.gen_isl_ast();
    function0.gen_halide_stmt();
    function0.gen_halide_obj("build/generated_fct_test_59.o");
}

int main(int argc, char **argv)
{
    generate_function_1("scheduled_with_before_overwrite", 100);

    return 0;
}
