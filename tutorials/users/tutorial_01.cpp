#include <tiramisu/tiramisu.h>

/**
    Goal
    ----
    Write a simple Tiramisu expression that assigns 0 to the array buf.
    The Tiramisu expression would be equivalent to the following C code:

    for (int i = 0; i < N; i++)
	buf0[i] = 0;

    How to compile ?
    ----------------
    g++ ...

*/

int main(int argc, char **argv)
{
    // Let us assume that we have a C++ code that has an array buf.
    // We want to write a Tiramisu expression that assigns 0 to this array.
    int buf[100];

    // All C++ files that call the Tiramisu API need to include
    // the file tiramisu/tiramisu.h

    // Initialize the Tiramisu compiler.
    tiramisu::init();

    // Declare Tiramisu computations (called A) attached to the buffer buf.
    tiramisu::comp A(buf);

    // Declare an iterator i that we will use to iterate over the computations A.
    tiramisu::iter i;

    // Assign 0 to each computation A(i)
    A(i) = 0;

    // Compile and run the Tiramisu expression.
    tiramisu::run();

    return 0;
}
