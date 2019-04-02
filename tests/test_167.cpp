#include <tiramisu/debug.h>
#include <tiramisu/core.h>
#include <tiramisu/topo_map.h>
using namespace tiramisu;

TopoMap generate_topo_map() {

  // lstopo or hwloc-ls to get the PUs
  // These are physical PU mappings, not logical. 
  // Uses hyperthreading (what would the numbers be without HT?)
  Proc numa0_proc("numa0_proc", {proc_pair(0,11), proc_pair(24,35)});
  Proc numa1_proc("numa1_proc", {proc_pair(12,23), proc_pair(24,35)});

  Socket numa0("numa0", 0, {numa0_proc});
  Socket numa1("numa1", 1, {numa1_proc});

  Node lanka01("lanka01", {numa0, numa1});
  Node lanka02("lanka02", {numa0, numa1});
  Node lanka03("lanka03", {numa0, numa1});
  Node lanka04("lanka04", {numa0, numa1});

  Rank r0(0, lanka01, numa0); // Proc is implied
  Rank r1(1, lanka01, numa1);
  Rank r2(2, lanka02, numa0);
  Rank r3(3, lanka02, numa1);
  Rank r4(4, lanka03, numa0);
  Rank r5(5, lanka03, numa1);
  Rank r6(6, lanka04, numa0);
  Rank r7(7, lanka04, numa1);

  // Options
  // Rank(0)...
  // Rank(0,lanka01)...
  TopoMap topo_map({r0, r1, r2, r3, r4, r5, r6, r7});
  
  return topo_map;
}

void generate_function_1(std::string name) {
    global::set_default_tiramisu_options();

    function function0(std::move(name));

    var x("x"), y("y"), p("p"), q("q");
    computation input("{input[x,y]: 0<=x<100 and 0<=y<100}", expr(), false, p_int32 , &function0);
    computation comp1("{comp1[x,y]: 0<=x<100 and 0<=y<100}", input(x,y), true, p_int32, &function0);
    computation comp2("{comp2[x,y]: 0<=x<100 and 0<=y<100}", comp1(x,y), true, p_int32, &function0);
    xfer sr = computation::create_xfer("{send[q,x,y]: 0<=q<7 and 0<=x<100 and 0<=y<100}",
                                       "{recv[p,x,y]: 1<=p<8 and 0<=x<100 and 0<=y<100}",
                                       q+1, p-1, xfer_prop(p_int32, {MPI, NONBLOCK, ASYNC}),
                                       xfer_prop(p_int32, {MPI, NONBLOCK, ASYNC}), comp1(x,y), &function0);

    sr.s->tag_distribute_level(q);
    sr.r->tag_distribute_level(p);
    local_wait wait_send = local_wait(sr.s->operator()(q,x,y), xfer_prop(p_wait, {MPI}), &function0);
    local_wait wait_recv = local_wait(sr.r->operator()(p,x,y), xfer_prop(p_wait, {MPI}), &function0);
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
    buffer buff_wait_send("buff_wait_send", {100, 100}, p_wait, a_temporary, &function0);
    buffer buff_wait_recv("buff_wait_recv", {100, 100}, p_wait, a_temporary, &function0);
    input.set_access("{input[x,y]->buff_input[x,y]}");
    sr.r->set_access("{recv[q,x,y]->buff_temp[x,y]}");
    comp1.set_access("{comp1[x,y]->buff_temp[x,y]}");
    comp2.set_access("{comp2[x,y]->buff[x,y]}");
    sr.s->set_wait_access("{send[q,x,y]->buff_wait_send[x,y]}");
    sr.r->set_wait_access("{recv[q,x,y]->buff_wait_recv[x,y]}");

    function0.codegen({&buff_input, &buff}, "build/generated_fct_test_167.o");
    TopoMap topo_map = generate_topo_map();
    topo_map.print_mapping();
    topo_map.generate_run_script("build/test_167_run");
}

int main() {
  generate_function_1("dist_topo_mapping");
  
}
