#include <tiramisu/tiramisu.h>
#include "configure.h"


using namespace tiramisu;

int main(int argc, char **argv)
{
    tiramisu::init("heat2d_tiramisu");
    // -------------------------------------------------------
    // Layer I
    // ------------------------------------------------------- 
    constant height("height", HEIGHT), width("width", WIDTH);
    var y_in("y_in", 0, height), x_in("x_in", 0, width), x("x", 1, width - 1), y("y", 1, height - 1);
    float alpha = ALPHA, beta = BETA;

    //Inputs
    input input("input", {y_in, x_in}, p_float32);

    //Computations
    computation heat2d_init("heat2d_init", {y_in, x_in}, (float)0);
    computation heat2d("heat2d", {y, x}, p_float32);
    heat2d.set_expression(expr(alpha) * input(y, x) + expr(beta) * (input(y, x + 1) + input(y, x -1) + input(y + 1, x) + input(y - 1, x)));

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    heat2d.after(heat2d_init, computation::root);
    heat2d_init.parallelize(y_in);
    heat2d.parallelize(y);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    //Buffers
    buffer buff_input("buff_input", {height, width}, p_float32, a_input);
    buffer buff_heat2d("buff_heat2d", {HEIGHT, WIDTH}, p_float32, a_output);

    //Store inputs
    input.store_in(&buff_input);

    //Store computations
    heat2d_init.store_in(&buff_heat2d); 
    heat2d.store_in(&buff_heat2d);

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------  
    tiramisu::codegen({&buff_input, &buff_heat2d}, "./generated_fct_heat2d.o");

    return 0;
}
