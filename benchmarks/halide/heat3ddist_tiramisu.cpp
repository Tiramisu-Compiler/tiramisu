#include <tiramisu/tiramisu.h>
#include "wrapper_heat3ddist.h"
using namespace tiramisu;

int main(int argc, char **argv)
{
    init("heat3ddist");

    constant X("X",_X);
    constant Y("Y",_Y);
    constant Z("Z",_Z);
    constant TIME("TIME",_TIME);

    constant ALPHA("ALPHA",_ALPHA);
    constant BETA("BETA",_BETA);

    constant NbNodes("NbNodes",NODES);

    constant SIZE("SIZE",_Z/NODES);//size of data per node for dimension Z
    //for heat3d_init
    var x_in("x_in",0,X);
    var y_in("y_in",0,Y);
    var z_in("z_in",0,Z);
    var t_in("t_in",0,TIME+1);
    //for heat3d_c
    var x("x",1,X-1);
    var y("y",1,Y-1);
    var z("z",1,Z-1);
    var t("t",1,TIME+1);

    input data("data",{z_in,y_in,x_in},p_float32);
    //init all data
    computation heat3d_init("heat3d_init",{t_in,z_in,y_in,x_in},data(z_in,y_in,x_in));
    //kernel
    computation heat3dc("heat3dc",{t,z,y,x},p_float32);
    heat3dc.set_expression(
		heat3dc(t-1,z,y,x) +
		expr(o_mul, ALPHA,
			  heat3dc(t-1,z-1,y,x) - expr(o_mul,BETA,heat3dc(t-1,z,y,x)) + heat3dc(t-1,z+1,y,x)
			+ heat3dc(t-1,z,y-1,x) - expr(o_mul,BETA,heat3dc(t-1,z,y,x)) + heat3dc(t-1,z,y+1,x)
			+ heat3dc(t-1,z,y,x-1) - expr(o_mul,BETA,heat3dc(t-1,z,y,x)) + heat3dc(t-1,z,y,x+1)));

    //distributing
    var z0("z0"),z1("z1");
    data.split(z_in,_Z/NODES,z0,z1);
    heat3d_init.split(z_in,_Z/NODES,z0,z1);
    heat3dc.split(z,_Z/NODES,z0,z1);

    data.tag_distribute_level(z0);
    heat3d_init.tag_distribute_level(z0);
    heat3dc.tag_distribute_level(z0);

    data.drop_rank_iter(z0);
    heat3d_init.drop_rank_iter(z0);
    heat3dc.drop_rank_iter(z0);

    //get the pointer to the function
    var r("r"),s("s"),i("i"),j("j"),k("k");//for the view
    var z_v("z_v",0,Z/NODES+2);
        view v("in",{z_v,y_in,x_in,t_in},p_float32);//view on buffer wherer are stored heat3dc results
    function* heat3d=global::get_implicit_function();
    //send from the left
    xfer send_previous_left = computation::create_xfer(
    "[TIME,Y,X,Z,NbNodes,SIZE]->{border_send_pleft[t,s,k,j,i]: 0<=s<NbNodes-1 and SIZE<=k<SIZE+1 and 0<=j<Y and 0<=i<X and 1<=t<TIME+1}",
    "[TIME,Y,X,NbNodes]->{border_recv_pleft[t,r,k,j,i]: 1<=r<NbNodes and 0<=k<1 and 0<=j<Y and 0<=i<X and 1<=t<TIME+1}",
    s+1,
    r-1,
    xfer_prop(p_float32, {MPI, BLOCK, ASYNC}),
    xfer_prop(p_float32, {MPI, BLOCK, ASYNC}),
    v(t-1,k,j,i), heat3d);

    //send from the right
    xfer send_previous_right = computation::create_xfer(
    "[TIME,Y,X,NbNodes]->{border_send_pright[t,s,k,j,i]: 1<=s<NbNodes and 1<=k<2 and 0<=j<Y and 0<=i<X and 1<=t<TIME+1}",
    "[TIME,Y,X,Z,NbNodes,SIZE]->{border_recv_pright[t,r,k,j,i]: 0<=r<NbNodes-1 and SIZE+1<=k<SIZE+2 and 0<=j<Y and 0<=i<X and 1<=t<TIME+1}",
    s-1,
    r+1,
    xfer_prop(p_float32, {MPI, BLOCK, ASYNC}),
    xfer_prop(p_float32, {MPI, BLOCK, ASYNC}),
    v(t-1,k,j,i), heat3d);

    //distributing communication
    send_previous_left.s->tag_distribute_level(s);
    send_previous_left.r->tag_distribute_level(r);
    send_previous_right.s->tag_distribute_level(s);
    send_previous_right.r->tag_distribute_level(r);
    //Order computations and communication
    heat3d_init.then(*send_previous_right.s, computation::root)
    .then(*send_previous_right.r,t)
    .then(*send_previous_left.s,t)
    .then(*send_previous_left.r,t)
    .then(heat3dc,t);

    //buffers
    buffer b_in("b_in",{SIZE,Y,X},p_float32,a_input);
    buffer b_out("b_out",{TIME+1,SIZE+2,Y,X},p_float32,a_output);

    data.store_in(&b_in,{z_in,y_in,x_in});
    heat3d_init.store_in(&b_out,{t_in,z_in+1,y_in,x_in});
    heat3dc.store_in(&b_out,{t,z+1,y,x});

    v.store_in(&b_out);
    send_previous_left.r->set_access("{border_recv_pleft[t,r,k,j,i]->b_out[t-1,k,j,i]}");
    send_previous_right.r->set_access("{border_recv_pright[t,r,k,j,i]->b_out[t-1,k,j,i]}");
    //code generation
    codegen({&b_in,&b_out}, "build/generated_fct_heat3ddist.o");

    return 0;
}
