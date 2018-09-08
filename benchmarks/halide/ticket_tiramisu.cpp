#include <tiramisu/tiramisu.h>

#define N 1024

using namespace tiramisu;

int main(int argc, char* argv[])
{
    tiramisu::init("ticket_tiramisu");

    var i("i", 0, N);
    computation Out("Out", {i}, ((uint8_t) 0));

    buffer b_Out("Out", {N}, p_uint8, a_output);
    Out.store_in(&b_Out);

    tiramisu::codegen({&b_Out}, "build/generated_fct_ticket.o");

  return 0;
}

