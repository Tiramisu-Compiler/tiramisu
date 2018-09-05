#include <tiramisu/tiramisu.h>

#define NN 8192
#define MM 8192

using namespace tiramisu;

int main(int argc, char* argv[])
{
    tiramisu::init("edge_tiramisu");

    var i("i", 0, NN-2), j("j", 0, MM-2), c("c", 0, 3);
    input Img("Img", {c, j, i}, p_uint8);

    // Layer I

    /* Ring blur filter. */
    computation R("R", {c, j, i}, (Img(c, j, i)   + Img(c, j+1, i)   + Img(c, j+2, i)+
				   Img(c, j, i+1)                    + Img(c, j+2, i+1)+
				   Img(c, j, i+2) + Img(c, j+1, i+2) + Img(c, j+2, i+2))/((uint8_t) 8));

    /* Robert's edge detection filter. */
    computation Out("Out", {c, j, i}, (R(c, j+1, i+1)-R(c, j, i+2)) + (R(c, j+1, i+2)-R(c, j, i+1)));

    // Layer II
    Out.after(R, computation::root);
//    R.tile(i,j, 64, 64, i1, j1, i2, j2)
//    R.vectorize(j2, 64);
//    Out.split(i, 64);

    // Layer III
    buffer b_Img("Img", {NN, MM, 3}, p_uint8, a_input);
    buffer   b_R("R",   {NN, MM, 3}, p_uint8, a_output);

    Img.store_in(&b_Img);
    R.store_in(&b_R);
    Out.store_in(&b_Img);

    tiramisu::codegen({&b_Img, &b_R}, "build/generated_fct_edge.o");

  return 0;
}

