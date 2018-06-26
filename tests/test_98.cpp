#include <tiramisu/debug.h>
#include <tiramisu/core.h>
#include "wrapper_test_98.h"

#include <Halide.h>

using namespace tiramisu;

// Do a distributed reduction without any communication. Distribute across 10 ranks.

void generate_function_1(std::string name)
{
    tiramisu::global::set_default_tiramisu_options();
    tiramisu::global::set_loop_iterator_type(p_int32);

    tiramisu::function function0(std::move(name));

    var x("x"), x1("x1"), x2("x2"), y("y");
    computation input("{input[x,y]: 0<=x<1000 and 0<=y<100}", expr(), false, p_int32 , &function0);
    computation S0_init("{S0_init[x,y]: 0<=x<1000 and 0<=y<100}", input(x,y), true, p_int32, &function0);
    // Reduce over REDUC_ITERS iterations.
    computation S0("{S0[iter,x,y]: 0<=iter<" + std::to_string(REDUC_ITERS) + " and 0<=x<1000 and 0<=y<100}", S0_init(x,y) * 2, true, p_int32 , &function0);

    input.split(x, 100, x1, x2);
    S0_init.split(x, 100, x1, x2);
    S0.split(x, 100, x1, x2);

    // Even though all of the ranks do the same computation, we still need to distribute so that we can drop the
    // index that will be distributed. If we didn't call tag_distribute_level, we would have to manually get rid of that
    // index by modifying layer 1.
    input.tag_distribute_level(x1);
    S0_init.tag_distribute_level(x1);
    S0_init.drop_rank_iter(x1);
    S0.tag_distribute_level(x1);
    S0.drop_rank_iter(x1);

    S0_init.before(S0, computation::root);

    buffer buff_input("buff_input", {100, 100}, p_int32 , a_input, &function0);
    buffer buff_output("buff_output", {100, 100}, p_int32 , a_output, &function0);

    input.set_access("{input[x,y]->buff_input[x,y]}");
    S0_init.set_access("{S0_init[x,y]->buff_output[x,y]}");
    S0.set_access("{S0[r,x,y]->buff_output[x,y]}");

    function0.codegen({&buff_input, &buff_output}, "build/generated_fct_test_98.o");
}


int main(int argc, char **argv)
{
    generate_function_1("dist_comp_only");

    return 0;
}
