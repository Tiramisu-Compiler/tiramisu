#include <tiramisu/tiramisu.h>
#include "configure.h"

using namespace tiramisu;

int main(int argc, char **argv)
{
    tiramisu::init("rgbyuv420_tiramisu");

    // -------------------------------------------------------
    // Layer I
    // ------------------------------------------------------- 
    constant width("width", WIDTH), height("height", HEIGHT), channels("channels", CHANNELS);
    constant  y_part_width("y_part_width", width), y_part_height("y_part_height", height), u_part_width("_part_width", width/2), u_part_height("u_part_height", height/2), v_part_width("v_part_width", width/2), v_part_height("v_part_height", height/2);
    var y_part_y("y_part_y", 0, y_part_height), y_part_x("y_part_x", 0, y_part_width), u_part_y("u_part_y", 0, u_part_height), u_part_x("u_part_x", 0, u_part_width), v_part_y("v_part_y", 0, v_part_height), v_part_x("v_part_x", 0, v_part_width);
    var in_width("in_width", 0, width), in_height("in_height", 0, height), in_channels("in_channels", 0, channels);

    //inputs
    input input_img("input_img", {in_channels, in_height, in_width}, p_uint8);

    //computations
    computation y_part("y_part", {y_part_y, y_part_x}, p_uint8);
    y_part.set_expression((input_img(0, y_part_y, y_part_x)*66 + input_img(1, y_part_y, y_part_x)*129 +  input_img(2, y_part_y, y_part_x)*25 + 128)*1 + 16);

    computation u_part("u_part", {u_part_y, u_part_x}, p_uint8);
    u_part.set_expression(input_img(0, 2*u_part_y, 2*u_part_x)*-38 - input_img(1,2*u_part_y, 2*u_part_x)*74 + (input_img(2, 2*u_part_y, 2*u_part_x) + 128)*112 + 128);

    computation v_part("v_part", {v_part_y, v_part_x}, p_uint8);
    v_part.set_expression(input_img(0, 2*v_part_y, 2*v_part_x)*112 - input_img(1, 2*v_part_y, 2*v_part_x)*94 - (input_img(2, 2*v_part_y, 2*v_part_x) + 128)*18  + 128);


    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------  
    y_part.parallelize(y_part_y);
    u_part.parallelize(u_part_y);
    v_part.parallelize(v_part_y);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    //Buffers
    buffer input_img_buf("input_img_buf", {channels, height, width}, p_uint8, a_input);
    buffer y_part_buf("y_part_buf", {y_part_height, y_part_width}, p_uint8, a_output);
    buffer u_part_buf("u_part_buf", {u_part_height, u_part_width}, p_uint8, a_output);
    buffer v_part_buf("v_part_buf", {v_part_height, v_part_width}, p_uint8, a_output);

    //Store inputs
    input_img.store_in(&input_img_buf);

    //Store computations
    y_part.store_in(&y_part_buf);
    u_part.store_in(&u_part_buf);
    v_part.store_in(&v_part_buf);

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    tiramisu::codegen({&input_img_buf, &y_part_buf, &u_part_buf, &v_part_buf}, "./generated_fct_rgbyuv420.o");

    return 0;
}
