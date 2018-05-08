#include <tiramisu/debug.h>
#include <tiramisu/core.h>

#include <Halide.h>

using namespace tiramisu;

// Do a distributed reduction where all the data starts on one node. Do fan out and then fan in to/from node 0.
// Same as test_101, but does a full collapse on the sends and recvs.

void generate_function_1(std::string name)
{
    tiramisu::global::set_default_tiramisu_options();

    tiramisu::function function0(std::move(name));

    var k("k"), x("x"), x1("x1"), x2("x2"), y("y"), q("q"), p("p");
    computation input("{input[x,y]: 0<=x<1000 and 0<=y<100}", expr(), false, p_int32 , &function0);
    computation S0_init("{S0_init[x,y]: 0<=x<1000 and 0<=y<100}", input(x,y), true, p_int32, &function0);
    // Reduce over 10 iterations.
    computation S0("{S0[i,x,y]: 0<=i<10 and 0<=x<1000 and 0<=y<100}", S0_init(x,y) * 2, true, p_int32 , &function0);
    // fan out data by x
    constant ONE("ONE", 1, p_int32, true, nullptr, 0, &function0);
    xfer fan_out = computation::create_xfer("[ONE]->{fan_out_s[p,q,x,y]: 0<=p<ONE and 1<=q<10 and 0<=x<100 and 0<=y<100}",
                                            "{fan_out_r[q,x,y]: 1<=q<10 and 0<=x<100 and 0<=y<100}",
                                            q, 0, xfer_prop(p_int32, {MPI, BLOCK, ASYNC}),
                                            xfer_prop(p_int32, {MPI, BLOCK, ASYNC}),
                                            input(q * 100 + x, y), &function0);
    xfer fan_in = computation::create_xfer("{fan_in_s[q,x,y]: 1<=q<10 and 0<=x<10 and 0<=y<100}",
                                           "[ONE]->{fan_in_r[p,q,x,y]: 0<=p<ONE and 1<=q<10 and 0<=x<100 and 0<=y<100}",
                                           0, q, xfer_prop(p_int32, {MPI, BLOCK, ASYNC}),
                                           xfer_prop(p_int32, {MPI, BLOCK, ASYNC}),
                                           input(x, y), &function0);


    input.split(x, 100, x1, x2);
    S0_init.split(x, 100, x1, x2);
    S0.split(x, 100, x1, x2);

    // collapse
    fan_out.s->collapse_many({collapse_group(3, 0, -1, 100), collapse_group(2, 0, -1, 100)});
    fan_out.r->collapse_many({collapse_group(2, 0, -1, 100), collapse_group(1, 0, -1, 100)});
    fan_in.s->collapse_many({collapse_group(2, 0, -1, 100), collapse_group(1, 0, -1, 100)});
    fan_in.r->collapse_many({collapse_group(3, 0, -1, 100), collapse_group(2, 0, -1, 100)});

    // Even though all of the ranks do the same computation, we still need to distribute so that we can drop the
    // index that will be distributed. If we didn't call tag_distribute_level, we would have to manually get rid of that
    // index by modifying layer 1.
    input.tag_distribute_level(x1);
    S0_init.tag_distribute_level(x1);
    S0_init.drop_rank_iter(x1);
    S0.tag_distribute_level(x1);
    S0.drop_rank_iter(x1);
    fan_out.s->tag_distribute_level(p);
    fan_out.r->tag_distribute_level(q);
    fan_in.s->tag_distribute_level(q);
    fan_in.r->tag_distribute_level(p);

    fan_out.s->before(*fan_out.r, computation::root);
    fan_out.r->before(S0_init, computation::root);
    S0_init.before(S0, computation::root);
    S0.before(*fan_in.s, computation::root);
    fan_in.s->before(*fan_in.r, computation::root);

    buffer buff_input("buff_input", {tiramisu::expr(o_select, var("rank") == 0, 1000, 100), 100},
                      p_int32 , a_input, &function0);
    buffer buff_output("buff_output", {tiramisu::expr(o_select, var("rank") == 0, 1000, 100), 100},
                       p_int32 , a_output, &function0);

    input.set_access("{input[x,y]->buff_input[x,y]}");
    S0_init.set_access("{S0_init[x,y]->buff_output[x,y]}");
    S0.set_access("{S0[i,x,y]->buff_output[x,y]}");
    fan_out.r->set_access("{fan_out_r[q,x,y]->buff_input[x,y]}");
    fan_in.r->set_access("{fan_in_r[p,q,x,y]->buff_output[q*100+x,y]}");

    function0.set_arguments({&buff_input, &buff_output});
    function0.lift_dist_comps();
    function0.gen_time_space_domain();
    function0.gen_isl_ast();
    function0.gen_halide_stmt();
    function0.gen_halide_obj("build/generated_fct_test_102.o");
}


int main(int argc, char **argv)
{
    generate_function_1("dist_reduction_collapse");

    return 0;
}

