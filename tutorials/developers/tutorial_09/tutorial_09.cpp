/**
 * A more complicated reduction.
 *
 * We want to represent the following program
 *
 * for y = 0 to 19
 *   for x = 0 to 9
 *     f(y, x) = 1
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

#include <tiramisu/tiramisu.h>
#include "wrapper_tutorial_09.h"

using namespace tiramisu;

void generate_function(std::string name, int size, int val0)
{
    // Set default tiramisu options.
    tiramisu::init();

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    function function0(name);

    var x = var("x");
    var y = var("y");
    var rx = var("rx");

    computation f("{f[y,x]: 0<=y<19 and 0<=x<9}", expr((uint8_t) 1), true, p_uint8, &function0);
    computation g_0("{g_0[y,-1]: 0<=y<19}",           expr((uint8_t) 0), true, p_uint8, &function0);
    computation g_1("{g_1[y,rx]: 0<=y<19 and 0<=rx<9}", g_0(y,rx-1) + f(y,rx), true, p_uint8, &function0);

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    g_0.after(f, computation::root);
    g_1.after(g_0, computation::root);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    buffer f_buff("f_buff", {19,size}, p_uint8, a_temporary, &function0);
    buffer g_buff("g_buff", {size}, p_uint8, a_output, &function0);
    f.set_access("{f[y,x]->f_buff[y,x]}");
    g_0.set_access("{g_0[y,rx]->g_buff[y]}");
    g_1.set_access("{g_1[y,rx]->g_buff[y]}");

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    function0.codegen({&g_buff}, "build/generated_fct_developers_tutorial_09.o");
}

int main(int argc, char **argv)
{
    generate_function("tiramisu_generated_code", SIZE1, 0);

    return 0;
}
