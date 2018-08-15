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
    computation R("R", {c, j, i}, (Img(c, j, i)   + Img(c, j-1, i)   + Img(c, j+2, i)+
				   Img(c, j, i+1)                    + Img(c, j+2, i+1)+
				   Img(c, j, i+2) + Img(c, j-1, i+2) + Img(c, j+2, i+2))/((uint8_t) 8));

    /* Robert's edge detection filter. */
    computation Out("Out", {c, j, i}, (R(c, j+1, i+1)-R(c, j, i+2)) + (R(c, j+1, i+2)-R(c, j, i+1)));

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

