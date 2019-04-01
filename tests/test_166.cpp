#include <tiramisu/debug.h>
#include <tiramisu/core.h>

#include <Halide.h>

using namespace tiramisu;
// test non-blocking communication
void generate_function_1(std::string name) {
    global::set_default_tiramisu_options();

    function function0(std::move(name));

    var x("x"), y("y"), p("p"), q("q");
    computation input("{input[x,y]: 0<=x<100 and 0<=y<100}", expr(), false, p_int32 , &function0);
    computation comp1("{comp1[x,y]: 0<=x<100 and 0<=y<100}", input(x,y), true, p_int32, &function0);
    computation comp2("{comp2[x,y]: 0<=x<100 and 0<=y<100}", comp1(x,y), true, p_int32, &function0);
    xfer sr = computation::create_xfer("{send[q,x,y]: 0<=q<9 and 0<=x<100 and 0<=y<100}",
                                       "{recv[p,x,y]: 1<=p<10 and 0<=x<100 and 0<=y<100}",
                                       q+1, p-1, xfer_prop(p_int32, {MPI, NONBLOCK, ASYNC}),
                                       xfer_prop(p_int32, {MPI, NONBLOCK, ASYNC}), comp1(x,y), &function0);

    sr.s->tag_distribute_level(q);
    sr.r->tag_distribute_level(p);
    tiramisu::wait wait_send(sr.s->operator()(q,x,y), xfer_prop(p_wait_ptr, {MPI}), &function0);
    tiramisu::wait wait_recv(sr.r->operator()(p,x,y), xfer_prop(p_wait_ptr, {MPI}), &function0);
    wait_send.tag_distribute_level(q);
    wait_recv.tag_distribute_level(p);
    
    // don't need to distribute comp for this example. Not very realistic, but that's not what
    // we are testing here.

    comp1.before(*sr.s, computation::root);
    sr.s->before(*sr.r, computation::root);
    sr.r->before(wait_recv, computation::root);
    wait_recv.before(comp2, computation::root);
    comp2.before(wait_send, computation::root);

    buffer buff_input("buff_input", {100, 100}, p_int32 , a_input, &function0);
    buffer buff_temp("buff_temp", {100, 100}, p_int32 , a_temporary, &function0);
    buffer buff("buff", {100, 100}, p_int32 , a_output, &function0);
    buffer buff_wait_send("buff_wait_send", {100, 100}, p_wait_ptr, a_temporary, &function0);
    buffer buff_wait_recv("buff_wait_recv", {100, 100}, p_wait_ptr, a_temporary, &function0);
    input.set_access("{input[x,y]->buff_input[x,y]}");
    sr.r->set_access("{recv[q,x,y]->buff_temp[x,y]}");
    comp1.set_access("{comp1[x,y]->buff_temp[x,y]}");
    comp2.set_access("{comp2[x,y]->buff[x,y]}");
    sr.s->set_wait_access("{send[q,x,y]->buff_wait_send[x,y]}");
    sr.r->set_wait_access("{recv[q,x,y]->buff_wait_recv[x,y]}");

    function0.codegen({&buff_input, &buff}, "build/generated_fct_test_166.o");
}

int main() {
    generate_function_1("dist_nonblocking");
    return 0;
}

