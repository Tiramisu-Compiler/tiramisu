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

void generate_function(std::string name, int size, int val0)
{
    tiramisu::global::set_default_tiramisu_options();

    tiramisu::function function0(name);
    tiramisu::constant N("N", tiramisu::expr((int32_t) size), p_int32, true, NULL, 0, &function0);
    tiramisu::constant M("M", tiramisu::expr((int32_t) size/2), p_int32, true, NULL, 0, &function0);

    tiramisu::buffer buf0("buf0", 2, {size,size}, tiramisu::p_uint8, NULL, a_output, &function0);

    tiramisu::expr e0 = tiramisu::expr((uint8_t) val0);
    tiramisu::computation S0("[N,M]->{S0[i,j]: 0<=i<N and 0<=j<N}", e0, true, p_uint8, &function0);

    tiramisu::idx i = tiramisu::idx("i");
    tiramisu::idx j = tiramisu::idx("j");
    tiramisu::expr e1 = tiramisu::expr(S0(i+i+0, j+j+0));
    tiramisu::computation S1("[N,M]->{S1[i,j]: 0<=i<M and 0<=j<M}", e1, true, p_uint8, &function0);

    tiramisu::expr e2 = tiramisu::expr(S0(2*i-i*2+2*i,1+2*j-1));
    tiramisu::computation S2("[N,M]->{S2[i,j]: 0<=i<M and 0<=j<M}", e2, true, p_uint8, &function0);

    tiramisu::expr e3 = tiramisu::expr(S0(0,-0));
    tiramisu::computation S3("[N,M]->{S3[i,j]: 0<=i<M and 0<=j<M}", e3, true, p_uint8, &function0);

    S0.set_access("[N,M]->{S0[i,j]->buf0[i,j]: 0<=i<N and 0<=j<N}");
    S1.set_access("[N,M]->{S1[i,j]->buf0[i,j]: 0<=i<M and 0<=j<M}");
    S2.set_access("[N,M]->{S2[i,j]->buf0[i,j]: 0<=i<M and 0<=j<M}");
    S3.set_access("[N,M]->{S3[i,j]->buf0[i,j]: 0<=i<M and 0<=j<M}");

    S1.after(S0, computation::root_dimension);
    S2.after(S1, computation::root_dimension);
    S3.after(S2, computation::root_dimension);

    function0.set_arguments({&buf0});
    function0.gen_time_processor_domain();
    function0.gen_isl_ast();
    function0.gen_halide_stmt();
    function0.gen_c_code();
    function0.gen_halide_obj("build/generated_fct_test_16.o");
}

int main(int argc, char **argv)
{
    generate_function("test_access_parsing", 10, 1);

    return 0;
}
