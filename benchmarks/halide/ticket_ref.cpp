#include "Halide.h"

using namespace Halide;

int main(int argc, char* argv[])
{
    Func Out("Out");
    Var i("i");

    Out(i) = ((uint8_t) 0);

    Out.compile_to_object("build/generated_fct_ticket_ref.o", {}, "ticket_ref");

  return 0;
}

