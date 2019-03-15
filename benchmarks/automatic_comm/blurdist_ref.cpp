#include <Halide.h>
#include "../include/tiramisu/core.h"
#include "wrapper_blurdist.h"

using namespace tiramisu;

int main(int argc, char **argv) {

    global::set_default_tiramisu_options();

    var i("i"), j("j"), i0("i0"), i1("i1"), ii("ii"), jj("jj"), p("p"), q("q");

    function blur("blurdist_ref");
    blur.add_context_constraints("[ROWS]->{:ROWS = "+std::to_string(_ROWS)+"}");

    constant ROWS("ROWS", expr((int32_t) _ROWS), p_int32, true, nullptr, 0, &blur);
    constant COLS("COLS", expr((int32_t) _COLS), p_int32, true, nullptr, 0, &blur);

    computation c_input("[ROWS,COLS]->{c_input[i,j]: 0<=i<ROWS and 0<=j<COLS+2}", expr(), false, p_uint32, &blur);

    expr e1 = (c_input(i, j) + c_input(i + 1, j) + c_input(i + 2, j)) / ((uint32_t) 3);

    computation c_blurx("[ROWS,COLS]->{c_blurx[i,j]: 0<=i<ROWS and 0<=j<COLS+2}", e1, true, p_uint32, &blur);

    expr e2 = (c_blurx(i, j) + c_blurx(i, j + 1) + c_blurx(i, j + 2)) / ((uint32_t) 3);

    computation c_blury("[ROWS,COLS]->{c_blury[i,j]: 0<=i<ROWS and 0<=j<COLS}", e2, true, p_uint32, &blur);

    c_input.split(i, _ROWS/10, i0, i1);
    c_blurx.split(i, _ROWS/10, i0, i1);
    c_blury.split(i, _ROWS/10, i0, i1);

    c_input.tag_distribute_level(i0);
    c_blurx.tag_distribute_level(i0);
    c_blury.tag_distribute_level(i0);

    c_input.drop_rank_iter(i0);
    c_blurx.drop_rank_iter(i0);
    c_blury.drop_rank_iter(i0);

    xfer border_comm = computation::create_xfer(
            "[COLS]->{border_send[p,ii,jj]: 1<=p<10 and 0<=ii<2 and 0<=jj<COLS+2}",
            "[COLS]->{border_recv[q,ii,jj]: 0<=q<9 and 0<=ii<2 and 0<=jj<COLS+2}",
            p-1,
            q+1,
            xfer_prop(p_uint32, {MPI, BLOCK, ASYNC}),
            xfer_prop(p_uint32, {MPI, BLOCK, ASYNC}),
            c_input(ii, jj), &blur);

    border_comm.s->tag_distribute_level(p);
    border_comm.r->tag_distribute_level(q);

    border_comm.s->before(*border_comm.r, computation::root);
    border_comm.r->before(c_blurx, computation::root);
    c_blurx.before(c_blury, i0);

    buffer b_input("b_input", {tiramisu::expr(_ROWS/10) + 2, tiramisu::expr(_COLS) + 2}, p_uint32, a_input, &blur);
    buffer b_blurx("b_blurx", {tiramisu::expr(_ROWS/10), tiramisu::expr(_COLS) + 2}, p_uint32, a_temporary, &blur);
    buffer b_blury("b_blury", {tiramisu::expr(_ROWS/10), tiramisu::expr(_COLS)}, p_uint32, a_output, &blur);
    border_comm.r->set_access("{border_recv[q,ii,jj]->b_input[" + std::to_string(_ROWS/10) + "+ii,jj]}");

    c_input.set_access("{c_input[i,j]->b_input[i,j]}");
    c_blurx.set_access("{c_blurx[i,j]->b_blurx[i,j]}");
    c_blury.set_access("{c_blury[i,j]->b_blury[i,j]}");



    blur.codegen({&b_input, &b_blury}, "build/generated_fct_blurdist_ref.o");

}
