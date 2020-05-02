#include <tiramisu/tiramisu.h>
#include "configure.h"


using namespace tiramisu;

int main(int argc, char **argv)
{
    tiramisu::init("cvtcolor_tiramisu");

    // -------------------------------------------------------
    // Layer I
    // ------------------------------------------------------- 
    constant width("width", WIDTH), height("height", HEIGHT), channels("channels", 3);
    var x("x", 0, width), y("y", 0, height), c("c", 0, channels);

    //inputs
    input input_img("input_img", {c, y, x}, p_int32);

    //Computations

    //RGB2Gray[y,x] <- (input_img[2,y,x]*1868 + input_img[1,y,x]*9617 + input_img[0,y,x]*4899 + 8192) / 16384
    computation RGB2Gray("RGB2Gray", {y, x}, cast(p_uint8, (input_img(2, y, x) * 1868 +  input_img(1, y, x) * 9617 + input_img(0, y, x) * 4899 + 8192) / 16384));
    
    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
    RGB2Gray.parallelize(y);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    //Buffers
    buffer input_buf("input_buf", {channels, height, width}, p_uint8, a_input);
    buffer RGB2Gray_buf("RGB2Gray_buf", {height, width}, p_uint8, a_output);

    //Store inputs
    input_img.store_in(&input_buf);

    //Store computations
    RGB2Gray.store_in(&RGB2Gray_buf);
 
    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    tiramisu::codegen({&input_buf, &RGB2Gray_buf}, "./generated_fct_cvtcolor.o");

    return 0;
}

