#include <Halide.h>
#include "../include/tiramisu/core.h"
#include "wrapper_blurxyautodist.h"

using namespace tiramisu;

int main(int argc, char **argv) {

    global::set_default_tiramisu_options();

    var i("i"), j("j"), i0("i0"), i1("i1");

    function blurxy("blurxydist_tiramisu");
    blurxy.add_context_constraints("[ROWS]->{:ROWS = "+std::to_string(_ROWS)+"}");

    constant ROWS("ROWS", expr((int32_t) _ROWS), p_int32, true, nullptr, 0, &blurxy);
    constant COLS("COLS", expr((int32_t) _COLS), p_int32, true, nullptr, 0, &blurxy);

    computation c_input("[ROWS,COLS]->{c_input[i,j]: 0<=i<ROWS and 0<=j<COLS+2}", expr(), false, p_uint32, &blurxy);

    expr e1 = (c_input(i, j) + c_input(i + 1, j) + c_input(i + 2, j)) / ((uint32_t) 3);

    computation c_blurx("[ROWS,COLS]->{c_blurx[i,j]: 0<=i<ROWS and 0<=j<COLS+2}", e1, true, p_uint32, &blurxy);

    expr e2 = (c_blurx(i, j) + c_blurx(i, j + 1) + c_blurx(i, j + 2)) / ((uint32_t) 3);

    computation c_blury("[ROWS,COLS]->{c_blury[i,j]: 0<=i<ROWS and 0<=j<COLS}", e2, true, p_uint32, &blurxy);

    c_input.split(i, _ROWS/10, i0, i1);
    c_blurx.split(i, _ROWS/10, i0, i1);
    c_blury.split(i, _ROWS/10, i0, i1);

    c_input.tag_distribute_level(i0);
    c_blurx.tag_distribute_level(i0);
    c_blury.tag_distribute_level(i0);

    c_input.drop_rank_iter(i0);
    c_blurx.drop_rank_iter(i0);
    c_blury.drop_rank_iter(i0);

    c_blurx.before(c_blury, i0);

    buffer b_input("b_input", {tiramisu::expr(_ROWS/10) , tiramisu::expr(_COLS) + 2}, p_uint32, a_input, &blurxy);
    buffer b_blurx("b_blurx", {tiramisu::expr(_ROWS/10), tiramisu::expr(_COLS) + 2}, p_uint32, a_temporary, &blurxy);
    buffer b_blury("b_blury", {tiramisu::expr(_ROWS/10), tiramisu::expr(_COLS)}, p_uint32, a_output, &blurxy);

    c_input.set_access("{c_input[i,j]->b_input[i,j]}");
    c_blurx.set_access("{c_blurx[i,j]->b_blurx[i,j]}");
    c_blury.set_access("{c_blury[i,j]->b_blury[i,j]}");
    c_blurx.gen_communication();

    blurxy.codegen({&b_input, &b_blury}, "build/generated_fct_blurxydist.o");

}
