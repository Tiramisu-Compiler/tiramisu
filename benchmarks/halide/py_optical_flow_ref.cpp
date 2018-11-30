#include "Halide.h"

int main(int argc, char* argv[])
{
    Halide::Func Res;
    Halide::Var x;

    Res(x) = 0;

    Res.compile_to_object("build/generated_fct_py_optical_flow_ref.o", {}, "py_optical_flow_ref");

    return 0;
}
