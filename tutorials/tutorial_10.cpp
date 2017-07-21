#include <isl/set.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>

#include <tiramisu/debug.h>
#include <tiramisu/core.h>

#include <string.h>
#include <Halide.h>
#include "../include/tiramisu/core.h"

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
    buf1[i, j] = buf0[i,j-1);

Output 1

Parallel for (int i = 0; i < 10; i++)
  for (int j = 0; j < 10; j++)
    buf0[i, j] = 3;
  for (int j = 1; j < 10; j++)
    buf1[i, j] = buf0[i,j-1];

Output 2

Parallel for (int i = 0; i < 10; i++)
  for (int j = 0; j < 10; j++)
    buf0[i, j] = 3;
    if (j < 9)
        buf1[i, j+1] = buf0[i,j];

*/

int main(int argc, char **argv)
{
    global::set_default_tiramisu_options();

    function function0("function0");

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    var i("i"), j("j");
    computation S0("[N]->{S0[i,j]: 0<=i<10 and 0<=j<10}", expr((uint8_t) 3), true, p_uint8, &function0);
    computation S1("[N]->{S1[i,j]: 0<=i<10 and 1<=j<10}", S0(i,j-1) + expr((uint8_t) 4), true, p_uint8, &function0);

    function0.dump_iteration_domain();

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

#if 1
    S1.after(S0, computation::root_dimension);
#elif 0
    S1.after(S0, 0);
    S0.tag_parallel_level(0);
#elif 0
    S1.shift(1, -1);
    S1.after(S0, 1);
    S0.tag_parallel_level(0);
#elif 0
    S1.shift(1, -1);
    S0.tag_parallel_level(0);
    S1.tile(0,1, 2,2);
    S0.tile(0,1, 2,2);
    S1.after(S0, computation::root_dimension);
#elif 0
    S1.shift(1, -1);
    S0.tag_parallel_level(0);
    S1.tile(0,1, 2,2);
    S0.tile(0,1, 2,2);
    S0.apply_transformation_on_schedule("{S0[i0, i1, i2, i3, i4, i5, i6, i7, i8, i9]->S0[i0, i1, i2, i3, i4, i5, i6, i7, i8, 0]}");
    S1.apply_transformation_on_schedule("{S1[i0, i1, i2, i3, i4, i5, i6, i7, i8, i9]->S1[i0, i1, i2, i3, i4, i5, i6, i7, i8, 1]}");
#elif 0
    S1.shift(1, -1);
    S0.tag_parallel_level(0);
    S1.tile(0,1, 2,2);
    S0.tile(0,1, 2,2);
    S0.tag_vector_level(3);
    S0.apply_transformation_on_schedule("{S0[i0, i1, i2, i3, i4, i5, i6, i7, i8, i9]->S0[i0, i1, i2, i3, i4, i5, i6, 0, i8, i9]}");
    S1.apply_transformation_on_schedule("{S1[i0, i1, i2, i3, i4, i5, i6, i7, i8, i9]->S1[i0, i1, i2, i3, i4, i5, i6, 1, i8, i9]}");
#endif

    function0.dump_schedule();

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

#if 1
    buffer buf0("buf0", 2, {tiramisu::expr(10), tiramisu::expr(10)}, p_uint8, NULL, a_temporary, &function0);
    S0.set_access("{S0[i,j]->buf0[i,j]}");
#elif 0
    buffer buf0("buf0", 1, {tiramisu::expr(1)}, p_uint8, NULL, a_temporary, &function0);
    S0.set_access("{S0[i,j]->buf0[0]}");
#elif 0
    buffer buf0("buf0", 1, {tiramisu::expr(2)}, p_uint8, NULL, a_temporary, &function0);
    S0.set_access("{S0[i,j]->buf0[j%2]}");
#endif

    buffer buf1("buf1", 2, {tiramisu::expr(10), tiramisu::expr(10)}, p_uint8, NULL, a_output, &function0);
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
    function0.gen_halide_obj("build/generated_fct_tutorial_10.o");

    return 0;
}