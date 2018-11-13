#include <tiramisu/tiramisu.h>
#include "wrapper_heat3d.h"
using namespace tiramisu;

int main(int argc, char **argv)
{
    init("heat3d_tiramisu");
    constant ROWS("ROWS", _X);
    constant COLS("COLS",  _Y);
    constant HEIGHT("HEIGHT",  _Z);
    constant TIME("TIME",_TIME);
    constant ALPHA("ALPHA",_ALPHA);
    constant BETA("BETA",_BETA);
    //for heat3d_init
    var x_in=var("x_in",0,ROWS);
    var y_in=var("y_in",0,COLS);
    var z_in=var("z_in",0,HEIGHT);
    var t_in=var("t_in",0,TIME+1);
    //for heat3d_c
    var x=var("x",1,ROWS-1);
    var y=var("y",1,COLS-1);
    var z=var("z",1,HEIGHT-1);
    var t=var("t",1,TIME+1);
    //input -- 3D
    input data("data",{z_in,y_in,x_in},p_float32);
    //init computation
    computation heat3d_init("heat3d_init",{t_in,z_in,y_in,x_in},data(z_in,y_in,x_in));
    //kernel
    computation heat3dc("heat3dc",{t,z,y,x},p_float32);
    heat3dc.set_expression(
		heat3dc(t-1,z,y,x) +
		expr(o_mul, ALPHA,
			  heat3dc(t-1,z-1,y,x) - expr(o_mul,BETA,heat3dc(t-1,z,y,x)) + heat3dc(t-1,z+1,y,x)
			+ heat3dc(t-1,z,y-1,x) - expr(o_mul,BETA,heat3dc(t-1,z,y,x)) + heat3dc(t-1,z,y+1,x)
			+ heat3dc(t-1,z,y,x-1) - expr(o_mul,BETA,heat3dc(t-1,z,y,x)) + heat3dc(t-1,z,y,x+1)));
    heat3dc.after(heat3d_init,computation::root); //we need to initialize all data before computing
    
    //buffers
    buffer b_in("b_in",{HEIGHT,COLS,ROWS},p_float32,a_input);
    buffer b_out("b_out",{TIME+1,HEIGHT,COLS,ROWS},p_float32,a_output);
    data.store_in(&b_in);
    heat3d_init.store_in(&b_out,{t_in,z_in,y_in,x_in});
    heat3dc.store_in(&b_out,{t,z,y,x});

    codegen({&b_in,&b_out}, "build/generated_fct_heat3d.o");
    return 0;
}
