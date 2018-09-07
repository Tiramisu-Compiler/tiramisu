#include <tiramisu/tiramisu.h>

#define NN 8192
#define MM 8192

using namespace tiramisu;

int main(int argc, char* argv[])
{
    tiramisu::init("edge_tiramisu");

    var i("i", 0, NN-2), j("j", 0, MM-2), c("c", 0, 3), i1("i1"), j1("j1"), i2("i2"), j2("j2");
    input Img("Img", {c, j, i}, p_uint8);

    // Layer I

    /* Ring blur filter. */
    computation R("R", {i, j, c}, (Img(i,   j, c) + Img(i,   j+1, c) + Img(i,   j+2, c)+
				   Img(i+1, j, c)                    + Img(i+1, j+2, c)+
				   Img(i+2, j, c) + Img(i+2, j+1, c) + Img(i+2, j+2, c))/((uint8_t) 8));

    /* Robert's edge detection filter. */
    computation Out("Out", {i, j, c}, (R(i+1, j+1, c)-R(i+2, j, c)) + (R(i+2, j+1, c)-R(i+1, j, c)));

    // Layer II
    Out.after(R, computation::root);
    R.tile(i, j, 64, 64, i1, j1, i2, j2);
    R.tag_parallel_level(i1);
    Out.tag_parallel_level(i);

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

