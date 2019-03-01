#include <tiramisu/debug.h>
#include <tiramisu/core.h>
#include "wrapper_test_141.h"

#include <Halide.h>
using namespace tiramisu;
void gen(std::string name)
{

    global::set_default_tiramisu_options();

    function boxblur(name);

    var i("i"), j("j"), i0("i0"), i1("i1");

    boxblur.add_context_constraints("[ROWS]->{: ROWS="+std::to_string(_ROWS)+"}");

    constant ROWS("ROWS", expr((int32_t) _ROWS), p_int32, true, nullptr, 0, &boxblur);
    constant COLS("COLS", expr((int32_t) _COLS), p_int32, true, nullptr, 0, &boxblur);

    computation img("[ROWS,COLS]->{img[i,j]: 0<=i<ROWS and 0<=j<COLS+2}", expr(), false, p_uint32, &boxblur);

    expr e1 = (img(i, j) + img(i + 1, j) + img(i + 2, j)) / ((uint32_t) 3);

    computation blurx("[ROWS,COLS]->{blurx[i,j]: 0<=i<ROWS and 0<=j<COLS+2}", e1, true, p_uint32, &boxblur);

    expr e2 = (blurx(i, j) + blurx(i, j + 1) + blurx(i, j + 2)) / ((uint32_t) 3);

    computation blury("[ROWS,COLS]->{blury[i,j]: 0<=i<ROWS and 0<=j<COLS}", e2, true, p_uint32, &boxblur);

    img.split(i, _ROWS/10, i0, i1);
    blurx.split(i, _ROWS/10, i0, i1);
    blury.split(i, _ROWS/10, i0, i1);

    img.tag_distribute_level(i0);
    blurx.tag_distribute_level(i0);
    blury.tag_distribute_level(i0);

    img.drop_rank_iter(i0);
    blurx.drop_rank_iter(i0);
    blury.drop_rank_iter(i0);

    blurx.before(blury, i0);

    buffer b_img("b_img", {tiramisu::expr(_ROWS/10), tiramisu::expr(_COLS) + 2}, p_uint32, a_input, &boxblur);
    buffer b_blurx("b_blurx", {tiramisu::expr(_ROWS/10), tiramisu::expr(_COLS) + 2}, p_uint32, a_temporary, &boxblur);
    buffer b_blury("b_blury", {tiramisu::expr(_ROWS/10), tiramisu::expr(_COLS)}, p_uint32, a_output, &boxblur);

    img.set_access("{img[i,j]->b_img[i,j]}");
    blurx.set_access("{blurx[i,j]->b_blurx[i,j]}");
    blury.set_access("{blury[i,j]->b_blury[i,j]}");

    blurx.gen_communication();

    boxblur.codegen({&b_img, &b_blury}, "build/generated_fct_test_" + std::string(TEST_NUMBER_STR) + ".o");
}

int main(int argc, char **argv)
{
    gen("boxblur");
    return 0;
}
