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

#include "wrapper_test_25.h"

using namespace tiramisu;

/**
 * Test update computations.
 *
 * The goal is to implement code equivalent to the following.
 *
 * C[0] = val0;
 * C[1] = val0;
 * C[i] = (C[i-1] + C[i-2])/2
 *
 */

void generate_function(std::string name, int size, int val0)
{
    tiramisu::global::set_default_tiramisu_options();

    tiramisu::function function0(name);
    tiramisu::constant N("N", tiramisu::expr((int32_t) size), p_int32, true, NULL, 0, &function0);

    tiramisu::buffer buf0("buf0", 1, {size}, tiramisu::p_uint8, NULL, a_output, &function0);

    tiramisu::var i = tiramisu::var("i");
    tiramisu::computation C("[N]->{C[0]}", tiramisu::expr((uint8_t) val0), true, p_uint8, &function0);

    tiramisu::computation *C1 = C.add_computations("[N]->{C[1]}", tiramisu::expr((uint8_t) val0), true,
                                p_uint8, &function0);
    tiramisu::computation *C2 = C.add_computations("[N]->{C[i]: 2<=i<N}",
                                (C(i - 1) + C(i - 2)) / ((uint8_t)2), true, p_uint8, &function0);

    C.set_access("[N]->{C[i]->buf0[i]}");
    C1->set_access("[N]->{C[i]->buf0[i]}");
    C2->set_access("[N]->{C[i]->buf0[i]}");

    C1->after(C, computation::root_dimension);
    C2->after((*C1), computation::root_dimension);

    function0.dump_dep_graph();
    function0.compute_bounds();

    function0.set_arguments({&buf0});
    function0.gen_time_space_domain();

    function0.dump(false);
    function0.dump_time_processor_domain();
    function0.dump_trimmed_time_processor_domain();

    function0.gen_isl_ast();
    function0.gen_halide_stmt();
    function0.gen_c_code();
    function0.gen_halide_obj("build/generated_fct_test_" + std::string(TEST_NUMBER_STR) + ".o");
}

int main(int argc, char **argv)
{
    generate_function("tiramisu_generated_code", SIZE0, 1);

    return 0;
}
