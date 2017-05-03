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

#include "wrapper_test_23.h"

using namespace tiramisu;

/**
 * Test computing dependence graph and bound inference. Part II.
 */

void generate_function(std::string name, int size, int val0)
{
    tiramisu::global::set_default_tiramisu_options();

    tiramisu::function function0(name);
    tiramisu::constant N("N", tiramisu::expr((int32_t) size), p_int32, true, NULL, 0, &function0);
    tiramisu::constant M("M", tiramisu::expr((int32_t) size), p_int32, true, NULL, 0, &function0);

    tiramisu::buffer buf0("buf0", 2, {size,size}, tiramisu::p_uint8, NULL, a_output, &function0);
    tiramisu::buffer buf1("buf1", 2, {size,size}, tiramisu::p_uint8, NULL, a_output, &function0);
    tiramisu::buffer buf2("buf2", 2, {size,size}, tiramisu::p_uint8, NULL, a_output, &function0);
    tiramisu::buffer buf3("buf3", 2, {size,size}, tiramisu::p_uint8, NULL, a_output, &function0);
    tiramisu::buffer buf4("buf4", 2, {size,size}, tiramisu::p_uint8, NULL, a_output, &function0);

    tiramisu::var i = tiramisu::var("i");
    tiramisu::var j = tiramisu::var("j");
    tiramisu::computation S0("[N,M]->{S0[i,j]                   }", tiramisu::expr((uint8_t) val0),        true, p_uint8, &function0);
    tiramisu::computation S1("[N,M]->{S1[i,j]: 0<=i<M and 0<=j<M}", S0(0,1),                               true, p_uint8, &function0);
    tiramisu::computation S2("[N,M]->{S2[i,j]                   }", S0(i,j) + S0(i,j),                     true, p_uint8, &function0);
    tiramisu::computation S3("[N,M]->{S3[i,j]                   }", S2(i,j) + S0(i,j),                     true, p_uint8, &function0);
    tiramisu::computation S4("[N,M]->{S4[i,j]: 0<=i<M and 0<=j<M}", S3(i,j) + tiramisu::expr((uint8_t) 1), true, p_uint8, &function0);

    S0.set_access("[N,M]->{S0[i,j]->buf0[i,j]                   }");
    S1.set_access("[N,M]->{S1[i,j]->buf1[i,j]: 0<=i<N and 0<=j<N}");
    S2.set_access("[N,M]->{S2[i,j]->buf2[i,j]                   }");
    S3.set_access("[N,M]->{S3[i,j]->buf3[i,j]                   }");
    S4.set_access("[N,M]->{S4[i,j]->buf4[i,j]: 0<=i<N and 0<=j<N}");

    S1.after(S0, computation::root_dimension);
    S2.after(S1, computation::root_dimension);
    S3.after(S2, computation::root_dimension);
    S4.after(S3, computation::root_dimension);

    // Compute the dep graph and dump it.
    function0.dump_dep_graph();
    function0.compute_bounds();

    function0.set_arguments({&buf0, &buf1, &buf2, &buf3, &buf4});
    function0.gen_time_processor_domain();
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
