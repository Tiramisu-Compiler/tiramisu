#include <tiramisu/tiramisu.h>
#include "wrapper_heat3d.h"
using namespace tiramisu;

int main(int argc, char **argv)
{
    tiramisu::init("heat3d_tiramisu");

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------     
    constant ROWS("ROWS", _X), COLS("COLS",  _Y), HEIGHT("HEIGHT",  _Z), TIME("TIME",_TIME), ALPHA("ALPHA",_ALPHA), BETA("BETA",_BETA);
    //for heat3d_init
    var x_in("x_in", 0, ROWS), y_in("y_in", 0, COLS), z_in("z_in", 0, HEIGHT), t_in("t_in", 0, TIME+1);
    //for heat3d_c
    var x("x",1,ROWS-1), y("y",1,COLS-1), z("z",1,HEIGHT-1), t("t",1,TIME+1);

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
  
    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------    
    heat3d_init.then(heat3dc, computation::root); //we need to initialize all data before computing
    heat3d_init.parallelize(t_in);
    
    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------    
    //buffers
    buffer b_in("b_in",{HEIGHT,COLS,ROWS},p_float32,a_input);
    buffer b_out("b_out",{TIME+1,HEIGHT,COLS,ROWS},p_float32,a_output);
    
    //Store inputs
    data.store_in(&b_in);

    //Store computations  
    heat3d_init.store_in(&b_out,{t_in,z_in,y_in,x_in});
    heat3dc.store_in(&b_out,{t,z,y,x});

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    tiramisu::codegen({&b_in,&b_out}, "./generated_fct_heat3d.o");
    return 0;
}
