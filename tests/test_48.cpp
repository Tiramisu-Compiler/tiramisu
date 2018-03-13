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

#include "wrapper_test_48.h"

using namespace tiramisu;

/**
 * Test S1.tag_parallel_level().
 */

void generate_function(std::string name, int size, int val0)
{
    tiramisu::global::set_default_tiramisu_options();
    tiramisu::global::set_loop_iterator_default_data_type(tiramisu::p_int32);

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    tiramisu::function function0(name);
    tiramisu::constant N("N", tiramisu::expr((int32_t) size), p_int32, true, NULL, 0, &function0);
    tiramisu::var i("i");
    tiramisu::var j("j");
    tiramisu::computation S0("[N]->{S0[i,j,k,l]: 0<=i<N and 0<=j<N and 0<=k<N and 0<=l<N}", tiramisu::expr((uint8_t) val0), true, p_uint8, &function0);
    tiramisu::computation S1("[N]->{S1[i,j,k,l]: 0<=i<N and 0<=j<N and 0<=k<N and 0<=l<N}", tiramisu::expr((uint8_t) (val0+1)), true, p_uint8, &function0);
    tiramisu::computation S2("[N]->{S2[i,j,k,l]: 0<=i<N and 0<=j<N and 0<=k<N and 0<=l<N}", tiramisu::expr((uint8_t) (val0+2)), true, p_uint8, &function0);
    tiramisu::computation S3("[N]->{S3[i,j,k,l]: 0<=i<N and 0<=j<N and 0<=k<N and 0<=l<N}", tiramisu::expr((uint8_t) (val0+3)), true, p_uint8, &function0);

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    S1.after(S0,computation::root);
    S2.after(S1,computation::root);
    S3.after(S2,computation::root);

    S0.tag_parallel_level(i);
    S1.tag_parallel_level(j);
    S2.tag_parallel_level(j);
    S3.tag_parallel_level(i);
    S3.tag_parallel_level(j);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    tiramisu::buffer buf0("buf0", {10, 10}, tiramisu::p_uint8, a_temporary, &function0);
    S0.set_access("[N,M]->{S0[i,j,k,l]->buf0[i,j]: 0<=i<N and 0<=j<N and 0<=k<N and 0<=l<N}");
    tiramisu::buffer buf1("buf1", {10, 10}, tiramisu::p_uint8, a_temporary, &function0);
    S1.set_access("[N,M]->{S1[i,j,k,l]->buf1[i,j]: 0<=i<N and 0<=j<N and 0<=k<N and 0<=l<N}");
    tiramisu::buffer buf2("buf2", {10, 10}, tiramisu::p_uint8, a_temporary, &function0);
    S2.set_access("[N,M]->{S2[i,j,k,l]->buf2[i,j]: 0<=i<N and 0<=j<N and 0<=k<N and 0<=l<N}");
    tiramisu::buffer buf3("buf3", {10, 10}, tiramisu::p_uint8, a_output, &function0);
    S3.set_access("[N,M]->{S3[i,j,k,l]->buf3[i,j]: 0<=i<N and 0<=j<N and 0<=k<N and 0<=l<N}");

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    function0.set_arguments({&buf3});
    function0.gen_time_space_domain();
    function0.gen_isl_ast();
    function0.gen_halide_stmt();
    function0.gen_halide_obj("build/generated_fct_test_" + std::string(TEST_NUMBER_STR) + ".o");
}

int main(int argc, char **argv)
{
    generate_function("tiramisu_generated_code", SIZE1, 2);

    return 0;
}
