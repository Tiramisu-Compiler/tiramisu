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

#define AUTOMATIC_VECTORIATION 1

using namespace tiramisu;

void generate_function_1(std::string name, int size, int val0)
{
    tiramisu::global::set_default_tiramisu_options();

    tiramisu::function function0(name);
    int Msize = 2*floor(size/2);
    tiramisu::constant N("N", tiramisu::expr((int32_t) size), p_int32, true, NULL, 0, &function0);
    tiramisu::constant M("M", tiramisu::expr((int32_t) Msize), p_int32, true, NULL, 0, &function0);

    tiramisu::buffer buf0("buf0", 2, {size,size}, tiramisu::p_uint8, NULL, a_output, &function0);

#if AUTOMATIC_VECTORIATION
    tiramisu::expr e1 = tiramisu::expr((uint8_t) val0);
    tiramisu::computation S0("[N,M]->{S0[i,j]: 0<=i<N and 0<=j<N}", e1, true, p_uint8, &function0);
    S0.set_access("[N,M]->{S0[i,j]->buf0[i,j]: 0<=i<N and 0<=j<N}");
    S0.vectorize(1, 2, tiramisu::var(p_int32, "N"));
#else
    function0.set_context_set("[N,M]->{:N>0 and M>0 and M%2=0 and M<=N}");

    tiramisu::expr e1 = tiramisu::expr((uint8_t) val0);
    tiramisu::expr e2 = tiramisu::expr((uint8_t) val0);

    tiramisu::computation S0("[N,M]->{S0[i,j]: 0<=i<N and 0<=j<M}", e1, true, p_uint8, &function0);
    tiramisu::computation S0P("[N,M]->{S0P[i,j]: 0<=i<N and M<=j<N}", e2, true, p_uint8, &function0);

    S0.set_access("[N,M]->{S0[i,j]->buf0[i,j]: 0<=i<N and 0<=j<N}");
    S0P.set_access("[N,M]->{S0P[i,j]->buf0[i,j]: 0<=i<N and 0<=j<N}");

    S0.split(3,2);
    S0.tag_vector_dimension(2);

    S0P.set_schedule("[N,M]->{S0P[i,j]->S0P[1,i,0,j,0,0,0]: 0<=i<N and M<=j<N}");
#endif

    function0.set_arguments({&buf0});
    function0.gen_time_processor_domain();
    function0.gen_isl_ast();
    function0.gen_halide_stmt();
    function0.gen_halide_obj("build/generated_fct_test_10.o");
}

int main(int argc, char **argv)
{
    generate_function_1("assign_7_to_10x10_2D_array_with_vectorization", 32, 7);

    return 0;
}
