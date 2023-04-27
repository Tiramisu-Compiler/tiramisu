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

#include "wrapper_test_95.h"

using namespace tiramisu;

/**
 * Test Google challenging bound inference.
 */

void generate_function(std::string name, int size, int val0)
{
    tiramisu::global::set_default_tiramisu_options();

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    tiramisu::function function0(name);
    tiramisu::constant N("N", tiramisu::expr((int32_t) size), p_int32, true, NULL, 0, &function0);
    tiramisu::constant M("M", tiramisu::expr((int32_t) size), p_int32, true, NULL, 0, &function0);
    tiramisu::var i("i"), j("j"), c("c");

//    tiramisu::computation f("[N,M]->{f[i0,i1,i2]: 0<=i0<=2 and 0<=i1<M and i2>=-1 and ((0<=i2<N) or (i2>=0 and 2i2<N) or 2i2<=-3+N)}", tiramisu::expr((float) val0), true, p_float32, &function0);
    tiramisu::computation f("[N,M]->{f[i0,i1,i2]: 0<=i0<=2 and 0<=i1<M and i2>=-1 and ((i2>=0 and 2i2<N) or 2i2<=-3+N)}", tiramisu::expr((float) val0), true, p_float32, &function0);
    tiramisu::computation upx("[N,M]->{upx[c,y,x]: 0<=c<3 and 0<=y<M and 0<=x<N}", tiramisu::expr((float) val0), true, p_float32, &function0);

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    upx.after(f, computation::root);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    tiramisu::buffer buf0("buf0", {SIZE, SIZE, 3}, tiramisu::p_float32, a_temporary, &function0);
    f.set_access("[N,M]->{f[c,y,x]->buf0[c,y,x]}");
    tiramisu::buffer buf1("buf1", {SIZE, SIZE, 3}, tiramisu::p_float32, a_output, &function0);
    upx.set_access("[N,M]->{upx[c,y,x]->buf1[c,y,x]}");

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    function0.set_arguments({&buf1});
    function0.gen_time_space_domain();
    function0.gen_isl_ast();
    function0.gen_halide_stmt();
    function0.gen_halide_obj("generated_fct_test_" + std::string(TEST_NUMBER_STR) + ".o");
}

int main(int argc, char **argv)
{
    generate_function("tiramisu_generated_code", SIZE, 1);

    return 0;
}
