#include <tiramisu/tiramisu.h>

#include <string.h>

#include "mttkrp_wrapper.h"

using namespace tiramisu;

/*
 * The goal is to generate code that implements the reference
 * mttkrp_ref.cpp
 */
void generate_function(std::string name, int size)
{
    tiramisu::init(name);

    constant N("N", size);

    var iB("iB", 0, N), kB("kB", 0, N), lB("lB", 0, N), jD("jD", 0, N);
    var iB0("iB0"), iB1("iB1"), kB0("kB0"), kB1("kB1"), lB0("lB0"), lB1("lB1");
    var jD0("jD0"), jD1("jD1"), jD10("jD10"), jD11("jD11");

    input B("B", {iB, kB, lB}, p_float64);
    input C("C", {kB, jD}, p_float64);
    input D("D", {lB, jD}, p_float64);

    computation A("A", {iB, jD}, expr((double) 0));
    computation A_update("A_update", {iB, kB, lB, jD}, A(iB, jD) + B(iB, kB, lB) * D(lB, jD) * C(kB, jD));

    global::get_implicit_function()->add_context_constraints("[N]->{:N="+std::to_string(SIZE)+"}");

    // -------------------------------------------------------
    // Scheduling
    // -------------------------------------------------------
    A.then(A_update, computation::root);

    // Tile sizes
    int B0 = 32, B1 = 64, B2 = 64, B3 = 32;

    int unrolling_factor = 16;
    A.tag_parallel_level(iB);
    A.vectorize(jD, B3);
    A_update.tile(iB, kB, lB, B0, B1, B2, iB0, kB0, lB0, iB1, kB1, lB1);
    A_update.tag_parallel_level(iB0);

    A_update.split(jD, B3, jD0, jD1);
    A_update.interchange(jD0, lB1);
    A_update.interchange(jD0, kB1);
    A_update.interchange(jD0, iB1);

    A_update.tag_unroll_level(lB1);

    A_update.tag_vector_level(jD1, B3);

    buffer buf_A("buf_A", {N, N}, p_float64, a_output);
    buffer buf_B("buf_B", {N, N, N}, p_float64, a_input);
    buffer buf_C("buf_C", {N, N}, p_float64, a_input);
    buffer buf_D("buf_D", {N, N}, p_float64, a_input);

    A.store_in(&buf_A);
    A_update.store_in(&buf_A, {iB, jD});
    B.store_in(&buf_B);
    C.store_in(&buf_C);
    D.store_in(&buf_D);
 
    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    tiramisu::codegen({&buf_A, &buf_B, &buf_C, &buf_D},
		      "generated_" + std::string(TEST_NAME_STR) + ".o");
}

int main(int argc, char **argv)
{
    generate_function("tiramisu_generated_code", SIZE);

    return 0;
}
