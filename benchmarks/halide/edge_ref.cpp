#include "Halide.h"

using namespace Halide;

int main(int argc, char* argv[])
{
    ImageParam Img(UInt(8), 3, "input");
    Func R("R"), Out("Out"), Img2("Img2");
    Var i("i"), j("j"), c("c");

    Img2(c, j, i) = BoundaryConditions::constant_exterior(Img, cast<uint8_t>(0))(c, j, i);

    /* Ring blur filter. */
    R(c, j, i) = (Img2(c, j, i)   + Img2(c, j+1, i)   + Img2(c, j+2, i)+
		  Img2(c, j, i+1)                     + Img2(c, j+2, i+1)+
		  Img2(c, j, i+2) + Img2(c, j+1, i+2) + Img2(c, j+2, i+2))/8;

    /* Robert's edge detection filter. */
    Out(c, j, i) = (R(c, j+1, i+1)-R(c, j, i+2)) + (R(c, j+1, i+2)-R(c, j, i+1));

    R.compute_root();
    Out.compute_root();

    Out.compile_to_object("build/generated_fct_edge_ref.o", {Img}, "edge_ref");
    Out.compile_to_lowered_stmt("build/generated_fct_edge_ref.txt", {Img}, Text);

  return 0;
}

