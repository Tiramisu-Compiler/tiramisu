#include <tiramisu/tiramisu.h>

using namespace tiramisu;

/*

 This tutorial shows an example of code and then shows
 different schedules for that code.


Input

for (int i = 0; i < 10; i++)
  for (int j = 0; j < 10; j++)
    buf0[i, j] = 3;
for (int i = 0; i < 10; i++)
  for (int j = 1; j < 10; j++)
    buf1[i, j] = buf0[i, j - 1];

Output 1

Parallel for (int i = 0; i < 10; i++)
  for (int j = 0; j < 10; j++)
    buf0[i, j] = 3;
  for (int j = 1; j < 10; j++)
    buf1[i, j] = buf0[i, j - 1];

Output 2

Parallel for (int i = 0; i < 10; i++)
  for (int j = 0; j < 10; j++)
    buf0[i, j] = 3;
    if (j < 9)
        buf1[i, j + 1] = buf0[i,j];

*/

int main(int argc, char **argv)
{
    // Set default tiramisu options.
    tiramisu::init();

    function function0("function0");

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    var i("i"), j("j"), i0("i0"), j0("j0"), i1("i1"), j1("j1");
    computation S0("[N]->{S0[i,j]: 0<=i<10 and 0<=j<10}", expr((uint8_t) 3), true, p_uint8, &function0);
    computation S1("[N]->{S1[i,j]: 0<=i<10 and 1<=j<10}", S0(i,j-1) + expr((uint8_t) 4), true, p_uint8, &function0);

    function0.dump_iteration_domain();

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------


#if 0
    S1.after(S0, computation::root);
#elif 0
    S1.after(S0, i);
    S0.tag_parallel_level(i);
#elif 0
    S1.shift(j, -1);
    S1.after(S0, j);
    S0.tag_parallel_level(i);
#elif 0
    S1.shift(j, -1);
    S0.tag_parallel_level(i);
    S1.tile(i,j, 2,2);
    S0.tile(i,j, 2,2);
    S1.after(S0, computation::root);
#elif 0
    S1.shift(j, -1);
    S0.tag_parallel_level(i);
    S1.tile(i,j, 2,2);
    S0.tile(i,j, 2,2);
    S0.apply_transformation_on_schedule("{S0[i0, i1, i2, i3, i4, i5, i6, i7, i8, i9]->S0[i0, i1, i2, i3, i4, i5, i6, i7, i8, 0]}");
    S1.apply_transformation_on_schedule("{S1[i0, i1, i2, i3, i4, i5, i6, i7, i8, i9]->S1[i0, i1, i2, i3, i4, i5, i6, i7, i8, 1]}");
#elif 0
    S1.shift(j, -1);
    S0.tag_parallel_level(i);
    S1.tile(i,j, 2,2);
    S0.tile(i,j, 2,2, i0,j0,i1,j1);
    S0.tag_vector_level(j1, 2);
    S0.apply_transformation_on_schedule("{S0[i0, i1, i2, i3, i4, i5, i6, i7, i8, i9]->S0[i0, i1, i2, i3, i4, i5, i6, 0, i8, i9]}");
    S1.apply_transformation_on_schedule("{S1[i0, i1, i2, i3, i4, i5, i6, i7, i8, i9]->S1[i0, i1, i2, i3, i4, i5, i6, 1, i8, i9]}");
#elif 1
    S1.shift(j, -1);
    S0.tag_parallel_level(i);
    S1.tile(i,j, 2,2, i0,j0,i1,j1);
    S0.tile(i,j, 2,2, i0,j0,i1,j1);
    S1.after(S0, j1);
#endif

    function0.dump_schedule();

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

#if 1
    buffer buf0("buf0", {expr(10), expr(10)}, p_uint8, a_temporary, &function0);
    S0.set_access("{S0[i,j]->buf0[i,j]}");
#elif 0
    buffer buf0("buf0", {expr(1)}, p_uint8, a_temporary, &function0);
    S0.set_access("{S0[i,j]->buf0[0]}");
#elif 0
    buffer buf0("buf0", {expr(2)}, p_uint8, a_temporary, &function0);
    S0.set_access("{S0[i,j]->buf0[j%2]}");
#endif

    buffer buf1("buf1", {expr(10), expr(10)}, p_uint8, a_output, &function0);
    S1.set_access("{S1[i,j]->buf1[i,j]}");

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    function0.set_arguments({&buf1});
    function0.gen_time_space_domain();
    function0.dump_time_processor_domain();
    function0.gen_isl_ast();
    function0.gen_halide_stmt();
    function0.dump_halide_stmt();
    function0.gen_halide_obj("build/generated_fct_developers_tutorial_10.o");

    return 0;
}
