#include <tiramisu/tiramisu.h>

#define NN 8192
#define MM 8192

using namespace tiramisu;

int main(int argc, char* argv[])
{
    tiramisu::init("edge_tiramisu");

    var i("i", 0, NN-2), j("j", 0, MM-2), c("c", 0, 3), i1("i1"), j1("j1"), i2("i2"), j2("j2"), c1("c1"), c2("c2");
    input Img("Img", {i, j, c}, p_uint8);

    buffer b_Img("Img", {3, MM, NN}, p_uint8, a_output);
    buffer   b_R("R",   {3, MM, NN}, p_uint8, a_temporary);
    buffer b_Img_gpu("Img_gpu", {3, MM, NN}, p_uint8, a_temporary);

    b_R.tag_gpu_global();
    b_Img_gpu.tag_gpu_global();

    // Layer I

    /* Ring blur filter. */
    computation R("R", {i, j, c}, (Img(i,   j, c) + Img(i,   j+1, c) + Img(i,   j+2, c)+
				   Img(i+1, j, c)                    + Img(i+1, j+2, c)+
				   Img(i+2, j, c) + Img(i+2, j+1, c) + Img(i+2, j+2, c))/((uint8_t) 8));

    /* Robert's edge detection filter. */
    computation Out("Out", {i, j, c}, (R(i+1, j+1, c)-R(i+2, j, c)) + (R(i+2, j+1, c)-R(i+1, j, c)));

    computation copy_Img_to_device({}, memcpy(b_Img, b_Img_gpu));
    computation copy_Img_to_host({}, memcpy(b_Img_gpu, b_Img));

    // Layer II
    R.gpu_tile(i, j, 8, 8, i1, j1, i2, j2);
    Out.gpu_tile(i, j, 8, 8, i1, j1, i2, j2);

    copy_Img_to_device.then(R, computation::root)
                      .then(Out, computation::root)
                      .then(copy_Img_to_host, computation::root);

    // Layer III

    Img.store_in(&b_Img_gpu, {c, j, i});
    R.store_in(&b_R, {c, j, i});
    Out.store_in(&b_Img_gpu, {c, j, i});

    tiramisu::codegen({&b_Img}, "build/generated_fct_edgegpu.o", true);

  return 0;
}

