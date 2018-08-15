#include <tiramisu/tiramisu.h>

#define NN 1024
#define MM 1024

using namespace tiramisu;

int main(int argc, char* argv[])
{
    tiramisu::init("edge_tiramisu");

    var i("i", 0, NN-2), j("j", 0, MM-2), c("c", 0, 3);
    input Img("Img", {i, j, c}, p_uint8);

    // Layer I

    /* Ring blur filter. */
    computation R("R", {i, j, c}, (Img(i,   j, c) + Img(i,   j-1, c) + Img(i,   j+2, c)+
				   Img(i+1, j, c)                    + Img(i+1, j+2, c)+
				   Img(i+2, j, c) + Img(i+2, j-1, c) + Img(i+2, j+2, c))/((uint8_t) 8));

    /* Robert's edge detection filter. */
    computation Out("Out", {i, j, c}, (R(i+1, j+1, c)-R(i+2, j, c)) + (R(i+2, j+1, c)-R(i+1, j, c)));

    // Layer II
    Out.after(R, computation::root);

    // Layer III
    buffer b_Img("Img", {NN, MM, 3}, p_uint8, a_input);
    buffer   b_R("R",   {NN, MM, 3}, p_uint8, a_temporary);
    buffer b_Out("Out", {NN, MM, 3}, p_uint8, a_output);

    Img.store_in(&b_Img);
    R.store_in(&b_R);
    Out.store_in(&b_Out);

    tiramisu::codegen({&b_Img, &b_Out}, "build/generated_fct_edge.o");

  return 0;
}

