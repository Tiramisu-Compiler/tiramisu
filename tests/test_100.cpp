#include <tiramisu/debug.h>
#include <tiramisu/core.h>

#include <Halide.h>

using namespace tiramisu;

// Verify that the recv can write into temporary buffers

void generate_function_1(std::string name) {
    global::set_default_tiramisu_options();

    function function0(std::move(name));

    var x("x"), y("y"), p("p"), q("q");
    computation input("{input[x,y]: 0<=x<100 and 0<=y<100}", expr(), false, p_int32 , &function0);
    computation comp1("{comp1[x,y]: 0<=x<100 and 0<=y<100}", input(x,y), true, p_int32, &function0);
    computation comp2("{comp2[x,y]: 0<=x<100 and 0<=y<100}", comp1(x,y), true, p_int32, &function0);
    xfer sr = computation::create_xfer("{send[q,x,y]: 0<=q<9 and 0<=x<100 and 0<=y<100}",
                                       "{recv[p,x,y]: 1<=p<10 and 0<=x<100 and 0<=y<100}",
                                       q+1, p-1, xfer_prop(p_int32, {MPI, BLOCK, ASYNC}),
                                       xfer_prop(p_int32, {MPI, BLOCK, ASYNC}), comp1(x,y), &function0);

    sr.s->tag_distribute_level(q);
    sr.r->tag_distribute_level(p);    
    // don't need to distribute comp for this example. Not very realistic, but that's not what
    // we are testing here.

    comp1.before(*sr.s, computation::root);
    sr.s->before(*sr.r, computation::root);
    sr.r->before(comp2, computation::root);

    buffer buff_input("buff_input", {100, 100}, p_int32 , a_input, &function0);
    buffer buff_temp("buff_temp", {100, 100}, p_int32 , a_temporary, &function0);
    buffer buff("buff", {100, 100}, p_int32 , a_output, &function0);
    input.set_access("{input[x,y]->buff_input[x,y]}");
    sr.r->set_access("{recv[q,x,y]->buff_temp[x,y]}");
    comp1.set_access("{comp1[x,y]->buff_temp[x,y]}");
    comp2.set_access("{comp2[x,y]->buff[x,y]}");

    function0.codegen({&buff_input, &buff}, "build/generated_fct_test_100.o");
}

int main() {
    generate_function_1("dist_temp_buffer");
    return 0;
}
