#include <tiramisu/tiramisu.h>

using namespace tiramisu;

int main(int argc, char* argv[])
{
    tiramisu::init("ticket_tiramisu");

    constant N("N", 1024);
    computation Out("[N]->{Out[i,j]: 0<=i<N and 0<=j<i}", ((uint8_t) 0), true, p_uint8, global::get_implicit_function());

    buffer b_Out("Out", {N,N}, p_uint8, a_output);
    Out.store_in(&b_Out);

    tiramisu::codegen({&b_Out}, "build/generated_fct_ticket.o");

  return 0;
}

