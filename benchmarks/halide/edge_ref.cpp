#include "Halide.h"

using namespace Halide;

int main(int argc, char* argv[])
{
    ImageParam Img2(UInt(8), 3, "Img2");
    Func R("R"), Out("Out");
    Var i("i"), j("j"), c("c");

    /* Ring blur filter. */
    R(i, j, c) = (Img2(i,   j, c) + Img2(i,   j+1, c) + Img2(i,   j+2, c)+
		  Img2(i+1, j, c)                     + Img2(i+1, j+2, c)+
		  Img2(i+2, j, c) + Img2(i+2, j+1, c) + Img2(i+2, j+2,
		      c))/((uint8_t) 8);

    /* Robert's edge detection filter. */
    Out(i, j, c) = (R(i+1, j+1, c)-R(i+2, j, c)) + (R(i+2, j+1, c)-R(i+1, j, c));

    R.compute_root();
    Out.compute_root();

    Out.compile_to_object("build/generated_fct_edge_ref.o", {Img2}, "edge_ref");
    Out.compile_to_lowered_stmt("build/generated_fct_edge_ref.txt", {Img2}, Text);

  return 0;
}

