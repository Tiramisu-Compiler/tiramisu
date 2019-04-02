#include <tiramisu/debug.h>
#include <tiramisu/core.h>
#include <tiramisu/topo_map.h>
using namespace tiramisu;

MultiTopo generate_topo_map() {

  // lstopo or hwloc-ls to get the PUs
  // These are physical PU mappings, not logical. 
  // Uses hyperthreading (what would the numbers be without HT?)
  Proc numa0_proc("numa0_proc", {proc_pair(0,11), proc_pair(24,35)});
  Proc numa1_proc("numa1_proc", {proc_pair(12,23), proc_pair(24,35)});
  Proc rand_proc("rand_proc", {proc_pair(14,20)});

  Socket numa0("numa0", 0, {numa0_proc});
  Socket numa1("numa1", 1, {numa1_proc});

  Node lanka01("lanka01", {numa0, numa1});
  Node lanka02("lanka02", {numa0, numa1});
  Node lanka03("lanka03", {numa0, numa1});
  Node lanka04("lanka04", {numa0, numa1});

  // Specifiy a pre-defined world_rank<->coord mapping. 
  // The user just has to specify how the coordinates should map and
  // tiramisu will automatically create the world rank mapping
  GridRank g00({0,0}, lanka01, numa0);
  GridRank g01({0,1}, lanka01, numa1);
  GridRank g10({1,0}, lanka02, numa0);
  GridRank g11({1,1}, lanka02, numa1);
  GridTopo *gr = new GridTopo({2,2}, {g00, g01, g10, g11});
  
  // Manually specify a linear (world) rank. In this case, we need an offset later on
  Rank r4(4, lanka03, numa0);
  Rank r5(5, lanka03, numa1); // TODO should be able to just specify cores without socket  
  Topo *topo = new Topo({r4, r5});

  MultiTopo topo_map({gr, topo});
  
  return topo_map;
}

void generate_function_1(std::string name) {
    global::set_default_tiramisu_options();

    function function0(std::move(name));

    var x("x"), y("y"), p("p"), q("q");
    var x1("x1"), x2("x2"), y1("y1"), y2("y2"); 
    computation input("{input[y,x]: 0<=x<100 and 0<=y<100}", expr(), false, p_int32 , &function0);
    computation input2("{input2[y,x]: 0<=x<100 and 0<=y<100}", expr(), false, p_int32 , &function0);
    computation comp1("{comp1[y,x]: 0<=x<100 and 0<=y<100}", input(y,x) * 17, true, p_int32, &function0);
    computation comp2("{comp2[y,x]: 0<=x<100 and 0<=y<100}", input2(y,x) - 22, true, p_int32, &function0);
    comp1.split(y, 50, y1, y2); 
    comp1.split(x, 50, x1, x2); 
    comp2.split(y, 50, y1, y2); 
    comp1.interchange(x1, y2);
    comp1.tag_distribute_level(y1);
    comp1.tag_distribute_level(x1);
    comp2.tag_distribute_level(y1, 4 /*rank offset*/, false);
    comp1.before(comp2, computation::root);

    buffer buff_input1("buff_input1", {50, 50}, p_int32 , a_input, &function0);
    buffer buff_input2("buff_input2", {50, 100}, p_int32 , a_input, &function0);
    buffer buff_out1("buff_out1", {50, 50}, p_int32 , a_output, &function0);
    buffer buff_out2("buff_out2", {50, 100}, p_int32 , a_output, &function0);
    input.set_access("[rank_dim_0, rank_dim_1]->{input[y,x]->buff_input1[y-(rank_dim_0*50),x-(rank_dim_1*50)]}");
    comp1.set_access("[rank_dim_0, rank_dim_1]->{comp1[y,x]->buff_out1[y-(rank_dim_0*50),x-(rank_dim_1*50)]}");
    input2.set_access("[rank_dim_0]->{input2[y,x]->buff_input2[y-(rank_dim_0*50),x]}");
    comp2.set_access("[rank_dim_0]->{comp2[y,x]->buff_out2[y-(rank_dim_0*50),x]}");

    MultiTopo topo_map = generate_topo_map();
    function0.codegen({&buff_input1, &buff_input2, &buff_out1, &buff_out2}, "build/generated_fct_test_167.o", topo_map);
    topo_map.print_mapping();
    topo_map.generate_run_script("build/test_167_run");
}

int main() {
  generate_function_1("dist_topo_mapping");  
}
