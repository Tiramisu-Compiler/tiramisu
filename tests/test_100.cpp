#include <tiramisu/debug.h>
#include <tiramisu/core.h>

#include <Halide.h>

using namespace tiramisu;

// Do just communication for distributed. Distribute across 10 ranks. All non-blocking/asynchronous.
// Overwrite each rank's first row with the received data

void generate_function_1(std::string name) {
    global::set_default_tiramisu_options();

    function function0(std::move(name));

    var x("x"), y("y"), q("q");
    computation input("{input[x,y]: 0<=x<1000 and 0<=y<100}", expr(), false, p_int32 , &function0);

    xfer sr = computation::create_xfer("{send[q,x,y]: 0<=q<9 and 0<=x<100 and 0<=y<100}",
                                       "{recv[q,x,y]: 1<=q<10 and 0<=x<100 and 0<=y<100}",
                                       q+1, q-1, xfer_prop(p_int32, {MPI, NONBLOCK, ASYNC}),
                                       xfer_prop(p_int32, {MPI, BLOCK, ASYNC}), input(x,y), &function0);

    // Use a constant so it doesn't remove this loop
    constant ONE("ONE", 1, p_int32, true, NULL, 0, &function0);
    constant NINE("NINE", 9, p_int32, true, NULL, 0, &function0);
    xfer sr2 = computation::create_xfer("[NINE]->{send2[q,x,y]: NINE<=q<10 and 0<=x<100 and 0<=y<100}",
                                        "[ONE]->{recv2[q,x,y]: 0<=q<ONE and 0<=x<100 and 0<=y<100}",
                                        0, 9, xfer_prop(p_int32, {MPI, BLOCK, ASYNC}),
                                        xfer_prop(p_int32, {MPI, BLOCK, ASYNC}), input(x,y), &function0);

    tiramisu::wait wait_send(sr.s->operator()(0,x,y), xfer_prop(p_wait_ptr, {MPI}), &function0);
    tiramisu::wait wait_recv(sr.r->operator()(0,x,y), xfer_prop(p_wait_ptr, {MPI}), &function0);
    tiramisu::wait wait_send2(sr2.s->operator()(0,x,y), xfer_prop(p_wait_ptr, {MPI}), &function0);
    tiramisu::wait wait_recv2(sr2.r->operator()(0,x,y), xfer_prop(p_wait_ptr, {MPI}), &function0);

    sr.s->tag_distribute_level(q);
    sr.r->tag_distribute_level(q);
    sr2.s->tag_distribute_level(q);
    sr2.r->tag_distribute_level(q);
    wait_send.tag_distribute_level(q);
    wait_recv.tag_distribute_level(q);
    wait_send2.tag_distribute_level(q);
    wait_recv2.tag_distribute_level(q);

    sr.s->before(*sr.r, computation::root);
    sr.r->before(wait_send, computation::root);
    wait_send.before(wait_recv, computation::root);
    wait_recv.before(*sr2.s, computation::root);
    sr2.s->before(*sr2.r, computation::root);
    sr2.r->before(wait_send2, computation::root);
    wait_send2.before(wait_recv2, computation::root);

    buffer buff("buff", {100, 100}, p_int32 , a_output, &function0);
    buffer buff_wait_send("buff_wait_send", {100, 100}, p_wait_ptr, a_temporary, &function0);
    buffer buff_wait_recv("buff_wait_recv", {100, 100}, p_wait_ptr, a_temporary, &function0);
    buffer buff_wait_send2("buff_wait_send2", {100, 100}, p_wait_ptr, a_temporary, &function0);
    buffer buff_wait_recv2("buff_wait_recv2", {100, 100}, p_wait_ptr, a_temporary, &function0);

    input.set_access("{input[x,y]->buff[x,y]}");
    // Overwrite the first row
    sr.r->set_access("{recv[q,x,y]->buff[0,y]}");
    sr2.r->set_access("{recv2[q,x,y]->buff[0,y]}");

    sr.s->set_wait_access("{send[q,x,y]->buff_wait_send[x,y]}");
    sr.r->set_wait_access("{recv[q,x,y]->buff_wait_recv[x,y]}");
    sr2.s->set_wait_access("{send2[q,x,y]->buff_wait_send2[x,y]}");
    sr2.r->set_wait_access("{recv2[q,x,y]->buff_wait_recv2[x,y]}");

    function0.codegen({&buff}, "build/generated_fct_test_100.o");
}

int main() {
    generate_function_1("dist_comm_only_nonblock");
    return 0;
}
