/**
  The goal of this tutorial is to implement in Tiramisu a code that is
  equivalent to the following

  for (int i = 0; i < 10; i++)
    for (int j = 0; j < 20; j++)
      output[i, j] = input[i, j] + i + 2;
*/

#include <tiramisu/tiramisu.h>

#define NN 10
#define MM 20

using namespace tiramisu;

int main(int argc, char **argv)
{
    // Set default tiramisu options.
    tiramisu::init();


    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    // Declare the function tut_02.
    function tut_02("tut_02");

    // Declare the constants N and M.
    constant N_const("N", expr((int32_t) NN), p_int32, true, NULL, 0, &tut_02);
    constant M_const("M", expr((int32_t) MM), p_int32, true, NULL, 0, &tut_02);

    // Declare variables
    var i("i"), j("j");

    // Declare a wrapper around the input.
    computation input("[N, M]->{input[i,j]: 0<=i<N and 0<=j<M}", expr(), false, p_uint8, &tut_02);

    // Declare expression and output computation.
    expr e = input(i, j) + cast(p_uint8, i) + (uint8_t)4;
    computation output("[N, M]->{output[i,j]: 0<=i<N and 0<=j<M}", e, true, p_uint8, &tut_02);



    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    // Set the schedule of the computation.
    var i0("i0"), i1("i1"), j0("j0"), j1("j1");
    output.tile(i, j, 2, 2, i0, j0, i1, j1);
    output.tag_parallel_level(i0);



    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    buffer b_input("b_input", {expr(NN), expr(MM)}, p_uint8, a_input, &tut_02);
    buffer b_output("b_output", {expr(NN), expr(MM)}, p_uint8, a_output, &tut_02);

    // Map the computations to a buffer.
    input.set_access("{input[i,j]->b_input[i,j]}");
    output.set_access("{output[i,j]->b_output[i,j]}");



    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    // Set the arguments to tut_02
    tut_02.set_arguments({&b_input, &b_output});
    // Generate code
    tut_02.gen_time_space_domain();
    tut_02.gen_isl_ast();
    tut_02.gen_halide_stmt();
    tut_02.gen_halide_obj("build/generated_fct_developers_tutorial_02.o");

    // Some debugging
    tut_02.dump_iteration_domain();
    tut_02.dump_halide_stmt();

    // Dump all the fields of the tut_02 class.
    tut_02.dump(true);

    return 0;
}
/**
 * Current limitations:
 * - Note that the type of the invariants N and M are "int32_t". This is
 *   important because these invariants are used later as loop bounds and the
 *   type of the bounds and the iterators should be the same for correct code
 *   generation. This implies that the invariants should be of type "int32_t".
 */
