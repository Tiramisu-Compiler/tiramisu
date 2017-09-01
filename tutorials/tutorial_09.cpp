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

#include "wrapper_tutorial_09.h"

using namespace tiramisu;

/**
 * A more complicated reduction.
 *
 * We want to represent the following program
 *
 * for y = 0 to 19
 *   for x = 0 to 9
 *     f(x, y) = 1
 *
 * for y = 0 to 19
 *   g(y) = 0
 * for y = 0 to 19
 *   for rx = 0 to 9
 *     g(y) += f(y, rx)
 *
 *
 * In order to implement this program, we create the following
 * computations
 *
 * {f[x,y]: 0<=x<19 and 0<=y<9}: 1
 * {g[y,-1]: 0<=y<19}: 0
 * {g[y,rx]: 0<=y<19 and 0<=rx<9}: g(y,rx-1) + f(y,rx)
 *
 */

void generate_function(std::string name, int size, int val0)
{
    tiramisu::global::set_default_tiramisu_options();


    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------


    tiramisu::function function0(name);

    tiramisu::var x = tiramisu::var("x");
    tiramisu::var y = tiramisu::var("y");
    tiramisu::var rx = tiramisu::var("rx");

    tiramisu::computation f("{f[y,x]: 0<=y<19 and 0<=x<9}", tiramisu::expr((uint8_t) 1), true, p_uint8, &function0);
    tiramisu::computation g("{g[y,-1]: 0<=y<19}",           tiramisu::expr((uint8_t) 0), true, p_uint8, &function0);
    tiramisu::computation *g2 = g.add_definitions("{g[y,rx]: 0<=y<19 and 0<=rx<9}", g(y,rx-1) + f(y,rx), true, p_uint8, &function0);


    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------


    g.after(f, computation::root_dimension);
    g2->after(g, computation::root_dimension);


    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------


    tiramisu::buffer f_buff("f_buff", 2, {19,size}, tiramisu::p_uint8, NULL, a_temporary, &function0);
    tiramisu::buffer g_buff("g_buff", 1, {size}, tiramisu::p_uint8, NULL, a_output, &function0);
    // Important: note that the access relations of the two computation C and C2 are identical.
    // The Tiramisu code generator assumes that the access relations of computations that have the same
    // name are identical.  In this case, the two relations are equal to "{C[j,i]->C_buff[i]}".
    f.set_access("{f[y,x]->f_buff[y,x]}");
    g.set_access("{g[y,rx]->g_buff[y]}");
    g2->set_access("{g[y,rx]->g_buff[y]}");



    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------


    function0.set_arguments({&g_buff});
    function0.gen_time_space_domain();
    function0.gen_isl_ast();
    function0.gen_halide_stmt();
    function0.gen_c_code();
    function0.gen_halide_obj("build/generated_fct_tutorial_09.o");
}

int main(int argc, char **argv)
{
    generate_function("tiramisu_generated_code", SIZE1, 0);

    return 0;
}
